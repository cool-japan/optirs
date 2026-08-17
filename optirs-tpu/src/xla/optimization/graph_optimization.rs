// Graph-level optimizations for XLA computations
//
// This module implements graph-level optimization passes: constant folding,
// algebraic simplification, common subexpression elimination and dead code
// elimination.
//
// Every pass in this module performs a real rewrite and reports whether it
// changed the graph. Passes that cannot do useful work on the current IR are
// not present at all rather than being registered as no-ops that still claim a
// speedup.

use std::fmt::Debug;

use scirs2_core::numeric::Float;
use std::collections::{HashMap, HashSet};

use super::super::frontend::{
    ConstantValue, OperandId, OperationId, OperationType, XLAComputation, XLAOperation,
};
use super::{OptimizationPass, OptimizationPipelineConfig};
use crate::error::Result;

/// Graph optimizer for XLA computations
pub struct GraphOptimizer<T: Float + Debug + Send + Sync + 'static> {
    /// Optimization configuration
    config: OptimizationPipelineConfig,

    /// Constant folding pass
    constant_folder: ConstantFoldingPass<T>,

    /// Dead code elimination pass
    dce_pass: DeadCodeEliminationPass<T>,

    /// Common subexpression elimination pass
    cse_pass: CommonSubexpressionEliminationPass<T>,

    /// Algebraic simplification pass
    algebraic_pass: AlgebraicSimplificationPass<T>,
}

/// Constant folding optimization pass
///
/// Evaluates operations whose operands are all `Constant` literals and replaces
/// them, in place, with the resulting literal.
pub struct ConstantFoldingPass<T: Float + Debug + Send + Sync + 'static> {
    /// Number of operations folded by the most recent run
    folded_count: usize,

    _phantom: std::marker::PhantomData<T>,
}

/// Dead code elimination pass
///
/// Keeps only the operations reachable backwards from the computation's
/// declared outputs.
pub struct DeadCodeEliminationPass<T: Float + Debug + Send + Sync + 'static> {
    /// Live operations set (retained between `mark` and `sweep`)
    live_operations: HashSet<OperationId>,

    /// Number of operations removed by the most recent run
    eliminated_count: usize,

    _phantom: std::marker::PhantomData<T>,
}

/// Common subexpression elimination pass
///
/// Replaces repeated evaluations of an identical, side-effect-free expression
/// with a single evaluation.
pub struct CommonSubexpressionEliminationPass<T: Float + Debug + Send + Sync + 'static> {
    /// Expression key to canonical operand mapping
    expression_map: HashMap<ExpressionKey, OperandId>,

    /// Eliminated expressions count
    eliminated_count: usize,

    _phantom: std::marker::PhantomData<T>,
}

/// Algebraic simplification pass
///
/// Applies identity and annihilator rewrites (`x*1`, `x+0`, `x*0`, ...).
pub struct AlgebraicSimplificationPass<T: Float + Debug + Send + Sync + 'static> {
    /// Number of rewrites applied by the most recent run
    applied_count: usize,

    _phantom: std::marker::PhantomData<T>,
}

/// Canonical key identifying a pure expression for CSE.
///
/// Commutative operations sort their operand ids so that `a + b` and `b + a`
/// hash and compare equal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ExpressionKey {
    op_type: OperationType,
    inputs: Vec<OperandId>,
}

/// Outcome of a single algebraic rewrite.
#[derive(Debug, Clone, PartialEq)]
enum AlgebraicRewrite {
    /// The operation is the identity on the given input; forward that operand.
    ForwardInput(OperandId),

    /// The operation always yields the given constant.
    ReplaceWithConstant(ConstantValue),
}

/// Operations whose evaluation is observable or non-deterministic and which
/// therefore must never be merged by CSE or removed as "redundant".
fn has_side_effects(op_type: &OperationType) -> bool {
    matches!(
        op_type,
        OperationType::Parameter
            | OperationType::Custom(_)
            | OperationType::Dropout
            | OperationType::Call
            | OperationType::While
            | OperationType::Conditional
            | OperationType::AllReduce(_)
            | OperationType::AllGather
            | OperationType::AllToAll
            | OperationType::CollectivePermute
            | OperationType::ReduceScatter
            | OperationType::Scatter
    )
}

/// True for operations whose operand order does not affect the result.
fn is_commutative(op_type: &OperationType) -> bool {
    matches!(
        op_type,
        OperationType::Add
            | OperationType::Multiply
            | OperationType::Maximum
            | OperationType::Minimum
            | OperationType::And
            | OperationType::Or
            | OperationType::Xor
            | OperationType::Equal
            | OperationType::NotEqual
    )
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> GraphOptimizer<T> {
    /// Create new graph optimizer
    pub fn new(config: &OptimizationPipelineConfig) -> Self {
        Self {
            config: config.clone(),
            constant_folder: ConstantFoldingPass::new(config),
            dce_pass: DeadCodeEliminationPass::new(config),
            cse_pass: CommonSubexpressionEliminationPass::new(),
            algebraic_pass: AlgebraicSimplificationPass::new(),
        }
    }

    /// Optimize computation graph.
    pub fn optimize(&mut self, computation: XLAComputation<T>) -> Result<XLAComputation<T>> {
        let (optimized, _changed) = self.optimize_tracked(computation)?;
        Ok(optimized)
    }

    /// Optimize the graph, reporting whether anything actually changed.
    ///
    /// The boolean drives honest reporting further up the pipeline: a pass that
    /// rewrote nothing must not be listed as applied.
    pub fn optimize_tracked(
        &mut self,
        computation: XLAComputation<T>,
    ) -> Result<(XLAComputation<T>, bool)> {
        let mut current = computation;
        let mut any_change = false;
        let mut iterations = 0usize;

        // Rewrites are mutually enabling (folding exposes algebraic identities,
        // which expose new common subexpressions), so run to a fixpoint.
        let max_iterations = if self.config.aggressive_mode { 16 } else { 8 };

        loop {
            iterations += 1;
            let mut changed = false;

            changed |= self.constant_folder.run(&mut current)?;
            changed |= self.algebraic_pass.run(&mut current)?;
            changed |= self.cse_pass.run(&mut current)?;

            any_change |= changed;

            if !changed || iterations >= max_iterations {
                break;
            }
        }

        // Dead code elimination runs once at the end, after every rewrite has
        // had a chance to orphan operations.
        any_change |= self.dce_pass.run(&mut current)?;

        Ok((current, any_change))
    }

    /// Number of operations removed by the most recent dead-code sweep.
    pub fn eliminated_operations(&self) -> usize {
        self.dce_pass.eliminated_count
    }
}

// ---------------------------------------------------------------------------
// Constant folding
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> ConstantFoldingPass<T> {
    /// Create new constant folding pass
    pub fn new(_config: &OptimizationPipelineConfig) -> Self {
        Self {
            folded_count: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Number of operations folded during the most recent run.
    pub fn folded_count(&self) -> usize {
        self.folded_count
    }

    /// Fold every foldable operation, returning whether the graph changed.
    fn run(&mut self, computation: &mut XLAComputation<T>) -> Result<bool> {
        self.folded_count = 0;

        // Repeat so that chains fold completely: folding `2+3` into `5` makes
        // `5*4` foldable on the next sweep.
        loop {
            let mut folded_this_sweep = 0usize;

            // Snapshot the constant operands so the borrow ends before mutation.
            let constants = Self::collect_constants(computation);

            let mut replacements: Vec<(OperationId, ConstantValue)> = Vec::new();
            for operation in &computation.operations {
                if matches!(operation.op_type, OperationType::Constant(_)) {
                    continue;
                }
                if let Some(value) = Self::evaluate(operation, &constants) {
                    replacements.push((operation.id, value));
                }
            }

            for (op_id, value) in replacements {
                if let Some(operation) = computation.operations.iter_mut().find(|op| op.id == op_id)
                {
                    operation.op_type = OperationType::Constant(value);
                    operation.inputs.clear();
                    folded_this_sweep += 1;
                }
            }

            if folded_this_sweep == 0 {
                break;
            }

            self.folded_count += folded_this_sweep;
            computation.rebuild_dependencies();
        }

        Ok(self.folded_count > 0)
    }

    /// Map every operand produced by a `Constant` operation to its literal.
    fn collect_constants(computation: &XLAComputation<T>) -> HashMap<OperandId, ConstantValue> {
        computation
            .operations
            .iter()
            .filter_map(|op| match &op.op_type {
                OperationType::Constant(value) => Some((op.output, value.clone())),
                _ => None,
            })
            .collect()
    }

    /// Evaluate an operation if all of its inputs are known constants.
    ///
    /// Arithmetic is carried out in `f64` because that is the storage type of
    /// [`ConstantValue`]; the graph element type only re-enters at execution.
    fn evaluate(
        operation: &XLAOperation<T>,
        constants: &HashMap<OperandId, ConstantValue>,
    ) -> Option<ConstantValue> {
        if operation.inputs.is_empty() {
            return None;
        }

        let mut operands: Vec<&ConstantValue> = Vec::with_capacity(operation.inputs.len());
        for input in &operation.inputs {
            operands.push(constants.get(input)?);
        }

        match (&operation.op_type, operands.as_slice()) {
            (OperationType::Add, [a, b]) => Self::binary(a, b, |x, y| x + y),
            (OperationType::Subtract, [a, b]) => Self::binary(a, b, |x, y| x - y),
            (OperationType::Multiply, [a, b]) => Self::binary(a, b, |x, y| x * y),
            (OperationType::Divide, [a, b]) => {
                // Refuse to fold a division by zero: materializing inf/NaN at
                // compile time changes observable behaviour.
                if b.data.contains(&0.0) {
                    None
                } else {
                    Self::binary(a, b, |x, y| x / y)
                }
            }
            (OperationType::Maximum, [a, b]) => Self::binary(a, b, f64::max),
            (OperationType::Minimum, [a, b]) => Self::binary(a, b, f64::min),
            (OperationType::Negate, [a]) => Self::unary(a, |x| -x),
            (OperationType::Abs, [a]) => Self::unary(a, f64::abs),
            (OperationType::Square, [a]) => Self::unary(a, |x| x * x),
            (OperationType::Exp, [a]) => Self::unary(a, f64::exp),
            (OperationType::Tanh, [a]) => Self::unary(a, f64::tanh),
            (OperationType::Sin, [a]) => Self::unary(a, f64::sin),
            (OperationType::Cos, [a]) => Self::unary(a, f64::cos),
            (OperationType::Ceil, [a]) => Self::unary(a, f64::ceil),
            (OperationType::Floor, [a]) => Self::unary(a, f64::floor),
            (OperationType::Round, [a]) => Self::unary(a, f64::round),
            (OperationType::Sqrt, [a]) => {
                if a.data.iter().any(|&v| v < 0.0) {
                    None
                } else {
                    Self::unary(a, f64::sqrt)
                }
            }
            (OperationType::Log, [a]) => {
                if a.data.iter().any(|&v| v <= 0.0) {
                    None
                } else {
                    Self::unary(a, f64::ln)
                }
            }
            _ => None,
        }
    }

    fn unary(value: &ConstantValue, f: impl Fn(f64) -> f64) -> Option<ConstantValue> {
        Some(ConstantValue {
            data: value.data.iter().copied().map(f).collect(),
            dims: value.dims.clone(),
        })
    }

    /// Elementwise binary evaluation with rank-0 scalar broadcasting.
    fn binary(
        lhs: &ConstantValue,
        rhs: &ConstantValue,
        f: impl Fn(f64, f64) -> f64,
    ) -> Option<ConstantValue> {
        if lhs.data.len() == rhs.data.len() {
            return Some(ConstantValue {
                data: lhs
                    .data
                    .iter()
                    .zip(rhs.data.iter())
                    .map(|(&a, &b)| f(a, b))
                    .collect(),
                dims: if lhs.dims.is_empty() {
                    rhs.dims.clone()
                } else {
                    lhs.dims.clone()
                },
            });
        }

        if let Some(scalar) = lhs.as_scalar() {
            return Some(ConstantValue {
                data: rhs.data.iter().map(|&b| f(scalar, b)).collect(),
                dims: rhs.dims.clone(),
            });
        }

        if let Some(scalar) = rhs.as_scalar() {
            return Some(ConstantValue {
                data: lhs.data.iter().map(|&a| f(a, scalar)).collect(),
                dims: lhs.dims.clone(),
            });
        }

        // Non-scalar shape mismatch: general broadcasting is not modelled here,
        // so decline to fold rather than guess.
        None
    }

    /// Count operations that could be folded right now.
    fn foldable_count(computation: &XLAComputation<T>) -> usize {
        let constants = Self::collect_constants(computation);
        computation
            .operations
            .iter()
            .filter(|op| !matches!(op.op_type, OperationType::Constant(_)))
            .filter(|op| Self::evaluate(op, &constants).is_some())
            .count()
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> OptimizationPass<T>
    for ConstantFoldingPass<T>
{
    fn name(&self) -> &str {
        "constant_folding"
    }

    fn apply(&mut self, mut computation: XLAComputation<T>) -> Result<XLAComputation<T>> {
        self.run(&mut computation)?;
        Ok(computation)
    }

    fn is_applicable(&self, computation: &XLAComputation<T>) -> bool {
        Self::foldable_count(computation) > 0
    }

    fn dependencies(&self) -> Vec<String> {
        vec![]
    }

    fn estimate_benefit(&self, computation: &XLAComputation<T>) -> f64 {
        if computation.operations.is_empty() {
            return 0.0;
        }
        Self::foldable_count(computation) as f64 / computation.operations.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Dead code elimination
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync>
    DeadCodeEliminationPass<T>
{
    /// Create new dead code elimination pass
    pub fn new(_config: &OptimizationPipelineConfig) -> Self {
        Self {
            live_operations: HashSet::new(),
            eliminated_count: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Number of operations removed by the most recent run.
    pub fn eliminated_count(&self) -> usize {
        self.eliminated_count
    }

    /// Mark live operations by walking backwards from the declared outputs.
    ///
    /// The walk is iterative so that long dependency chains cannot overflow the
    /// stack.
    fn mark_live_operations(&mut self, computation: &XLAComputation<T>) {
        self.live_operations.clear();

        let producers: HashMap<OperandId, OperationId> = computation
            .operations
            .iter()
            .map(|op| (op.output, op.id))
            .collect();
        let by_id: HashMap<OperationId, &XLAOperation<T>> = computation
            .operations
            .iter()
            .map(|op| (op.id, op))
            .collect();

        let mut worklist: Vec<OperationId> = Vec::new();

        for output in &computation.outputs {
            if let Some(&op_id) = producers.get(&output.operand) {
                worklist.push(op_id);
            }
        }

        // Side-effecting operations are roots in their own right: removing an
        // all-reduce because nothing reads its result would change semantics.
        for operation in &computation.operations {
            if has_side_effects(&operation.op_type) {
                worklist.push(operation.id);
            }
        }

        while let Some(op_id) = worklist.pop() {
            if !self.live_operations.insert(op_id) {
                continue;
            }
            if let Some(operation) = by_id.get(&op_id) {
                for input in &operation.inputs {
                    if let Some(&producer) = producers.get(input) {
                        if !self.live_operations.contains(&producer) {
                            worklist.push(producer);
                        }
                    }
                }
            }
        }
    }

    /// Remove unreachable operations, returning whether the graph changed.
    fn run(&mut self, computation: &mut XLAComputation<T>) -> Result<bool> {
        self.eliminated_count = 0;

        if computation.operations.is_empty() {
            return Ok(false);
        }

        // FAIL-SAFE: a computation with no declared outputs has no liveness
        // root, so a naive sweep would delete the entire graph. Refuse to run
        // instead, and say so loudly.
        if computation.outputs.is_empty() {
            log::warn!(
                "dead_code_elimination skipped for computation '{}': no outputs are declared, \
                 so liveness cannot be determined (call set_outputs or \
                 mark_terminal_operands_as_outputs)",
                computation.metadata.name
            );
            return Ok(false);
        }

        self.mark_live_operations(computation);

        let before = computation.operations.len();
        let live = std::mem::take(&mut self.live_operations);
        computation.operations.retain(|op| live.contains(&op.id));
        self.live_operations = live;

        self.eliminated_count = before.saturating_sub(computation.operations.len());
        if self.eliminated_count == 0 {
            return Ok(false);
        }

        // Drop operands no surviving operation references, but never drop an
        // operand a declared output points at.
        let mut used: HashSet<OperandId> = computation
            .operations
            .iter()
            .flat_map(|op| op.inputs.iter().chain(std::iter::once(&op.output)))
            .copied()
            .collect();
        used.extend(computation.outputs.iter().map(|o| o.operand));

        computation
            .operands
            .retain(|operand_id, _| used.contains(operand_id));

        computation.rebuild_dependencies();

        Ok(true)
    }

    /// Number of operations that are currently unreachable from the outputs.
    fn dead_count(&mut self, computation: &XLAComputation<T>) -> usize {
        if computation.outputs.is_empty() {
            return 0;
        }
        self.mark_live_operations(computation);
        computation
            .operations
            .iter()
            .filter(|op| !self.live_operations.contains(&op.id))
            .count()
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> OptimizationPass<T>
    for DeadCodeEliminationPass<T>
{
    fn name(&self) -> &str {
        "dead_code_elimination"
    }

    fn apply(&mut self, mut computation: XLAComputation<T>) -> Result<XLAComputation<T>> {
        self.run(&mut computation)?;
        Ok(computation)
    }

    fn is_applicable(&self, computation: &XLAComputation<T>) -> bool {
        !computation.operations.is_empty() && !computation.outputs.is_empty()
    }

    fn dependencies(&self) -> Vec<String> {
        vec![]
    }

    fn estimate_benefit(&self, computation: &XLAComputation<T>) -> f64 {
        if computation.operations.is_empty() {
            return 0.0;
        }
        // `estimate_benefit` takes `&self` by trait contract, so use a scratch
        // instance rather than mutating shared marking state.
        let mut probe = Self {
            live_operations: HashSet::new(),
            eliminated_count: 0,
            _phantom: std::marker::PhantomData,
        };
        probe.dead_count(computation) as f64 / computation.operations.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Common subexpression elimination
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> Default
    for CommonSubexpressionEliminationPass<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync>
    CommonSubexpressionEliminationPass<T>
{
    /// Create new CSE pass
    pub fn new() -> Self {
        Self {
            expression_map: HashMap::new(),
            eliminated_count: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Number of duplicate expressions removed by the most recent run.
    pub fn eliminated_count(&self) -> usize {
        self.eliminated_count
    }

    /// Canonical key for an operation, or `None` when it must not be merged.
    fn expression_key(operation: &XLAOperation<T>) -> Option<ExpressionKey> {
        if has_side_effects(&operation.op_type) {
            return None;
        }

        let mut inputs = operation.inputs.clone();
        if is_commutative(&operation.op_type) {
            inputs.sort_by_key(|operand| operand.0);
        }

        Some(ExpressionKey {
            op_type: operation.op_type.clone(),
            inputs,
        })
    }

    /// Eliminate duplicate expressions, returning whether the graph changed.
    fn run(&mut self, computation: &mut XLAComputation<T>) -> Result<bool> {
        self.expression_map.clear();
        self.eliminated_count = 0;

        // Operations are stored in dependency order, so a single forward pass
        // sees every definition before its uses.
        let mut duplicates: Vec<(OperationId, OperandId, OperandId)> = Vec::new();

        for operation in &computation.operations {
            let Some(key) = Self::expression_key(operation) else {
                continue;
            };

            match self.expression_map.get(&key) {
                Some(&canonical) => {
                    duplicates.push((operation.id, operation.output, canonical));
                }
                None => {
                    self.expression_map.insert(key, operation.output);
                }
            }
        }

        if duplicates.is_empty() {
            return Ok(false);
        }

        for (op_id, redundant_operand, canonical_operand) in duplicates {
            // Only merge when the two operands agree on shape; a differing
            // shape means the operands are not actually interchangeable.
            let shapes_match = match (
                computation.operands.get(&redundant_operand),
                computation.operands.get(&canonical_operand),
            ) {
                (Some(a), Some(b)) => a.shape == b.shape,
                _ => false,
            };
            if !shapes_match {
                continue;
            }

            computation.replace_operand_uses(redundant_operand, canonical_operand);
            computation.operations.retain(|op| op.id != op_id);
            computation.operands.remove(&redundant_operand);
            self.eliminated_count += 1;
        }

        if self.eliminated_count == 0 {
            return Ok(false);
        }

        computation.rebuild_dependencies();
        Ok(true)
    }

    /// Number of operations that are duplicates of an earlier expression.
    fn duplicate_count(computation: &XLAComputation<T>) -> usize {
        let mut seen: HashSet<ExpressionKey> = HashSet::new();
        let mut duplicates = 0usize;
        for operation in &computation.operations {
            if let Some(key) = Self::expression_key(operation) {
                if !seen.insert(key) {
                    duplicates += 1;
                }
            }
        }
        duplicates
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> OptimizationPass<T>
    for CommonSubexpressionEliminationPass<T>
{
    fn name(&self) -> &str {
        "common_subexpression_elimination"
    }

    fn apply(&mut self, mut computation: XLAComputation<T>) -> Result<XLAComputation<T>> {
        self.run(&mut computation)?;
        Ok(computation)
    }

    fn is_applicable(&self, computation: &XLAComputation<T>) -> bool {
        Self::duplicate_count(computation) > 0
    }

    fn dependencies(&self) -> Vec<String> {
        vec![]
    }

    fn estimate_benefit(&self, computation: &XLAComputation<T>) -> f64 {
        if computation.operations.is_empty() {
            return 0.0;
        }
        Self::duplicate_count(computation) as f64 / computation.operations.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Algebraic simplification
// ---------------------------------------------------------------------------

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> Default
    for AlgebraicSimplificationPass<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync>
    AlgebraicSimplificationPass<T>
{
    /// Create new algebraic simplification pass
    pub fn new() -> Self {
        Self {
            applied_count: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Number of rewrites applied during the most recent run.
    pub fn applied_count(&self) -> usize {
        self.applied_count
    }

    /// Determine the rewrite (if any) that applies to `operation`.
    ///
    /// Implemented rules:
    /// - `x * 1 = x`, `1 * x = x`
    /// - `x + 0 = x`, `0 + x = x`
    /// - `x - 0 = x`
    /// - `x / 1 = x`
    /// - `x * 0 = 0`, `0 * x = 0`
    fn match_rule(
        operation: &XLAOperation<T>,
        constants: &HashMap<OperandId, ConstantValue>,
    ) -> Option<AlgebraicRewrite> {
        let [lhs, rhs] = operation.inputs.as_slice() else {
            return None;
        };
        let (lhs, rhs) = (*lhs, *rhs);

        let lhs_const = constants.get(&lhs);
        let rhs_const = constants.get(&rhs);

        // Only rank-0/uniform literals participate: a tensor literal that
        // happens to contain a 1 in one position is not a multiplicative
        // identity.
        let is_one = |value: Option<&ConstantValue>| value.is_some_and(|c| c.is_uniform(1.0));
        let is_zero = |value: Option<&ConstantValue>| value.is_some_and(|c| c.is_uniform(0.0));

        match operation.op_type {
            OperationType::Multiply => {
                if is_zero(lhs_const) || is_zero(rhs_const) {
                    // Annihilator; produce a zero literal of the result shape.
                    return Some(AlgebraicRewrite::ReplaceWithConstant(
                        ConstantValue::scalar(0.0),
                    ));
                }
                if is_one(rhs_const) {
                    return Some(AlgebraicRewrite::ForwardInput(lhs));
                }
                if is_one(lhs_const) {
                    return Some(AlgebraicRewrite::ForwardInput(rhs));
                }
                None
            }
            OperationType::Add => {
                if is_zero(rhs_const) {
                    return Some(AlgebraicRewrite::ForwardInput(lhs));
                }
                if is_zero(lhs_const) {
                    return Some(AlgebraicRewrite::ForwardInput(rhs));
                }
                None
            }
            // Subtraction and division are not commutative, so only the
            // right-hand identity is valid.
            OperationType::Subtract if is_zero(rhs_const) => {
                Some(AlgebraicRewrite::ForwardInput(lhs))
            }
            OperationType::Divide if is_one(rhs_const) => Some(AlgebraicRewrite::ForwardInput(lhs)),
            _ => None,
        }
    }

    /// Apply all matching rewrites, returning whether the graph changed.
    fn run(&mut self, computation: &mut XLAComputation<T>) -> Result<bool> {
        self.applied_count = 0;

        loop {
            let constants = ConstantFoldingPass::<T>::collect_constants(computation);

            let mut forwards: Vec<(OperationId, OperandId, OperandId)> = Vec::new();
            let mut constant_rewrites: Vec<(OperationId, ConstantValue)> = Vec::new();

            for operation in &computation.operations {
                match Self::match_rule(operation, &constants) {
                    Some(AlgebraicRewrite::ForwardInput(source)) => {
                        // Forwarding is only sound when the forwarded operand
                        // has the same shape as the result (no implicit
                        // broadcast is being elided).
                        let compatible = match (
                            computation.operands.get(&source),
                            computation.operands.get(&operation.output),
                        ) {
                            (Some(a), Some(b)) => a.shape == b.shape,
                            _ => false,
                        };
                        if compatible {
                            forwards.push((operation.id, operation.output, source));
                        }
                    }
                    Some(AlgebraicRewrite::ReplaceWithConstant(value)) => {
                        let shaped = computation
                            .operands
                            .get(&operation.output)
                            .map(|operand| {
                                let count = operand.shape.element_count.max(1);
                                ConstantValue {
                                    data: vec![
                                        value.as_scalar().unwrap_or(0.0);
                                        if operand.shape.dimensions.is_empty() {
                                            1
                                        } else {
                                            count
                                        }
                                    ],
                                    dims: operand.shape.dimensions.clone(),
                                }
                            })
                            .unwrap_or_else(|| value.clone());
                        constant_rewrites.push((operation.id, shaped));
                    }
                    None => {}
                }
            }

            if forwards.is_empty() && constant_rewrites.is_empty() {
                break;
            }

            for (op_id, redundant, source) in forwards {
                computation.replace_operand_uses(redundant, source);
                computation.operations.retain(|op| op.id != op_id);
                computation.operands.remove(&redundant);
                self.applied_count += 1;
            }

            for (op_id, value) in constant_rewrites {
                if let Some(operation) = computation.operations.iter_mut().find(|op| op.id == op_id)
                {
                    operation.op_type = OperationType::Constant(value);
                    operation.inputs.clear();
                    self.applied_count += 1;
                }
            }

            computation.rebuild_dependencies();
        }

        Ok(self.applied_count > 0)
    }

    /// Number of operations a rewrite currently matches.
    fn match_count(computation: &XLAComputation<T>) -> usize {
        let constants = ConstantFoldingPass::<T>::collect_constants(computation);
        computation
            .operations
            .iter()
            .filter(|op| Self::match_rule(op, &constants).is_some())
            .count()
    }
}

impl<T: Float + Debug + Default + std::fmt::Debug + Clone + Send + Sync> OptimizationPass<T>
    for AlgebraicSimplificationPass<T>
{
    fn name(&self) -> &str {
        "algebraic_simplification"
    }

    fn apply(&mut self, mut computation: XLAComputation<T>) -> Result<XLAComputation<T>> {
        self.run(&mut computation)?;
        Ok(computation)
    }

    fn is_applicable(&self, computation: &XLAComputation<T>) -> bool {
        Self::match_count(computation) > 0
    }

    fn dependencies(&self) -> Vec<String> {
        vec![]
    }

    fn estimate_benefit(&self, computation: &XLAComputation<T>) -> f64 {
        if computation.operations.is_empty() {
            return 0.0;
        }
        Self::match_count(computation) as f64 / computation.operations.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::frontend::graph_capture::test_support::{add_op, scalar_shape, shape};
    use super::super::super::frontend::ComputationGraphBuilder;
    use super::super::{ComputeCapability, HardwareTarget, OptimizationPipeline};
    use super::*;

    fn pipeline_config() -> OptimizationPipelineConfig {
        OptimizationPipelineConfig {
            optimization_level: crate::main_types::XLAOptimizationLevel::Standard,
            enable_graph_optimization: true,
            enable_kernel_fusion: true,
            enable_memory_optimization: true,
            enable_scheduling_optimization: true,
            max_optimization_time: 300,
            target_hardware: HardwareTarget {
                tpu_version: "v4".to_string(),
                num_cores: 4,
                memory_capacity: 1024 * 1024 * 1024,
                memory_bandwidth: 1600.0,
                compute_capability: ComputeCapability {
                    matrix_unit_dims: (128, 128),
                    vector_unit_width: 256,
                    supported_dtypes: vec!["F32".to_string()],
                    special_instructions: vec![],
                },
            },
            custom_passes: vec![],
            aggressive_mode: false,
            debug_mode: false,
        }
    }

    #[test]
    fn test_constant_folding_pass_metadata() {
        let pass: ConstantFoldingPass<f32> = ConstantFoldingPass::new(&pipeline_config());
        assert_eq!(pass.name(), "constant_folding");
        assert_eq!(pass.dependencies().len(), 0);
    }

    /// F2/F19: `2 + 3` must fold to the literal `5`.
    #[test]
    fn constant_folding_evaluates_addition() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("fold");

        let two = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(2.0)),
            vec![],
            scalar_shape(),
        );
        let three = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(3.0)),
            vec![],
            scalar_shape(),
        );
        let sum = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![two, three],
            scalar_shape(),
        );
        builder
            .set_outputs(&mut comp, &[sum])
            .expect("outputs must be settable");

        let mut pass: ConstantFoldingPass<f32> = ConstantFoldingPass::new(&pipeline_config());
        let changed = pass.run(&mut comp).expect("folding must succeed");

        assert!(changed, "constant folding must report a change");
        let folded = comp
            .producer_of(sum)
            .expect("sum operand must still be produced");
        match &folded.op_type {
            OperationType::Constant(value) => assert_eq!(value.as_scalar(), Some(5.0)),
            other => panic!("expected a folded constant, got {other:?}"),
        }
        assert!(folded.inputs.is_empty(), "a literal has no operands");
    }

    /// Chained folding: `(2 + 3) * 4` collapses to `20` in one run.
    #[test]
    fn constant_folding_folds_chains() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("chain");

        let two = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(2.0)),
            vec![],
            scalar_shape(),
        );
        let three = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(3.0)),
            vec![],
            scalar_shape(),
        );
        let four = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(4.0)),
            vec![],
            scalar_shape(),
        );
        let sum = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![two, three],
            scalar_shape(),
        );
        let product = add_op(
            &mut builder,
            &mut comp,
            OperationType::Multiply,
            vec![sum, four],
            scalar_shape(),
        );
        builder
            .set_outputs(&mut comp, &[product])
            .expect("outputs must be settable");

        let mut pass: ConstantFoldingPass<f32> = ConstantFoldingPass::new(&pipeline_config());
        pass.run(&mut comp).expect("folding must succeed");

        match comp.producer_of(product).map(|op| &op.op_type) {
            Some(OperationType::Constant(value)) => assert_eq!(value.as_scalar(), Some(20.0)),
            other => panic!("expected folded constant 20, got {other:?}"),
        }
    }

    /// Division by zero must not be folded into an infinity at compile time.
    #[test]
    fn constant_folding_refuses_division_by_zero() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("div0");

        let one = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(1.0)),
            vec![],
            scalar_shape(),
        );
        let zero = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(0.0)),
            vec![],
            scalar_shape(),
        );
        let quotient = add_op(
            &mut builder,
            &mut comp,
            OperationType::Divide,
            vec![one, zero],
            scalar_shape(),
        );
        builder
            .set_outputs(&mut comp, &[quotient])
            .expect("outputs must be settable");

        let mut pass: ConstantFoldingPass<f32> = ConstantFoldingPass::new(&pipeline_config());
        let changed = pass.run(&mut comp).expect("folding must succeed");

        assert!(!changed, "division by zero must not be folded");
        assert!(matches!(
            comp.producer_of(quotient).map(|op| &op.op_type),
            Some(OperationType::Divide)
        ));
    }

    /// F1: DCE must never delete a graph whose outputs were never declared.
    #[test]
    fn dce_is_a_no_op_without_declared_outputs() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("no_outputs");

        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let _ = add_op(
            &mut builder,
            &mut comp,
            OperationType::Square,
            vec![a],
            scalar_shape(),
        );

        assert!(comp.outputs.is_empty());

        let mut pass: DeadCodeEliminationPass<f32> =
            DeadCodeEliminationPass::new(&pipeline_config());
        let changed = pass.run(&mut comp).expect("dce must succeed");

        assert!(!changed);
        assert_eq!(
            comp.operations.len(),
            2,
            "DCE must not delete the graph when liveness is undeterminable"
        );
    }

    /// DCE removes only what the outputs cannot reach.
    #[test]
    fn dce_removes_only_unreachable_operations() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("dce");

        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let live = add_op(
            &mut builder,
            &mut comp,
            OperationType::Square,
            vec![a],
            scalar_shape(),
        );
        // Nothing consumes this and it is not an output.
        let _dead = add_op(
            &mut builder,
            &mut comp,
            OperationType::Abs,
            vec![a],
            scalar_shape(),
        );

        builder
            .set_outputs(&mut comp, &[live])
            .expect("outputs must be settable");

        let mut pass: DeadCodeEliminationPass<f32> =
            DeadCodeEliminationPass::new(&pipeline_config());
        let changed = pass.run(&mut comp).expect("dce must succeed");

        assert!(changed);
        assert_eq!(comp.operations.len(), 2, "parameter and square survive");
        assert!(comp.producer_of(live).is_some());
        assert!(comp
            .operations
            .iter()
            .all(|op| !matches!(op.op_type, OperationType::Abs)));
    }

    /// F19: identical subexpressions are computed once.
    #[test]
    fn cse_merges_identical_expressions() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("cse");

        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let b = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let first = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![a, b],
            scalar_shape(),
        );
        // Same expression with the operands swapped: Add is commutative.
        let second = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![b, a],
            scalar_shape(),
        );
        let combined = add_op(
            &mut builder,
            &mut comp,
            OperationType::Multiply,
            vec![first, second],
            scalar_shape(),
        );
        builder
            .set_outputs(&mut comp, &[combined])
            .expect("outputs must be settable");

        let before = comp.operations.len();
        let mut pass: CommonSubexpressionEliminationPass<f32> =
            CommonSubexpressionEliminationPass::new();
        let changed = pass.run(&mut comp).expect("cse must succeed");

        assert!(changed, "the duplicated Add must be eliminated");
        assert_eq!(comp.operations.len(), before - 1);
        assert_eq!(pass.eliminated_count(), 1);

        // The surviving Multiply now reads the canonical operand twice.
        let mul = comp
            .producer_of(combined)
            .expect("product must survive CSE");
        assert_eq!(mul.inputs, vec![first, first]);
    }

    /// Constants with different payloads must not be merged by CSE.
    #[test]
    fn cse_distinguishes_different_constants() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("cse_const");

        let one = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(1.0)),
            vec![],
            scalar_shape(),
        );
        let two = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(2.0)),
            vec![],
            scalar_shape(),
        );
        let sum = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![one, two],
            scalar_shape(),
        );
        builder
            .set_outputs(&mut comp, &[sum])
            .expect("outputs must be settable");

        let mut pass: CommonSubexpressionEliminationPass<f32> =
            CommonSubexpressionEliminationPass::new();
        pass.run(&mut comp).expect("cse must succeed");

        assert_eq!(
            pass.eliminated_count(),
            0,
            "1.0 and 2.0 are different constants"
        );
    }

    /// Two identical constants, however, are one expression.
    #[test]
    fn cse_merges_equal_constants() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("cse_dup_const");

        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(7.0)),
            vec![],
            scalar_shape(),
        );
        let b = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(7.0)),
            vec![],
            scalar_shape(),
        );
        let sum = add_op(
            &mut builder,
            &mut comp,
            OperationType::Subtract,
            vec![a, b],
            scalar_shape(),
        );
        builder
            .set_outputs(&mut comp, &[sum])
            .expect("outputs must be settable");

        let mut pass: CommonSubexpressionEliminationPass<f32> =
            CommonSubexpressionEliminationPass::new();
        pass.run(&mut comp).expect("cse must succeed");

        assert_eq!(pass.eliminated_count(), 1);
    }

    /// F19: `x * 1` becomes `x`.
    #[test]
    fn algebraic_simplification_removes_multiply_by_one() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("mul_one");

        let x = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let one = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(1.0)),
            vec![],
            scalar_shape(),
        );
        let product = add_op(
            &mut builder,
            &mut comp,
            OperationType::Multiply,
            vec![x, one],
            scalar_shape(),
        );
        let result = add_op(
            &mut builder,
            &mut comp,
            OperationType::Square,
            vec![product],
            scalar_shape(),
        );
        builder
            .set_outputs(&mut comp, &[result])
            .expect("outputs must be settable");

        let mut pass: AlgebraicSimplificationPass<f32> = AlgebraicSimplificationPass::new();
        let changed = pass.run(&mut comp).expect("simplification must succeed");

        assert!(changed);
        let square = comp
            .producer_of(result)
            .expect("square must survive simplification");
        assert_eq!(
            square.inputs,
            vec![x],
            "the Square must now read x directly"
        );
    }

    /// `x + 0` becomes `x`.
    #[test]
    fn algebraic_simplification_removes_add_zero() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("add_zero");

        let x = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let zero = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(0.0)),
            vec![],
            scalar_shape(),
        );
        let sum = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![zero, x],
            scalar_shape(),
        );
        let result = add_op(
            &mut builder,
            &mut comp,
            OperationType::Abs,
            vec![sum],
            scalar_shape(),
        );
        builder
            .set_outputs(&mut comp, &[result])
            .expect("outputs must be settable");

        let mut pass: AlgebraicSimplificationPass<f32> = AlgebraicSimplificationPass::new();
        assert!(pass.run(&mut comp).expect("simplification must succeed"));

        let abs = comp.producer_of(result).expect("abs must survive");
        assert_eq!(abs.inputs, vec![x]);
    }

    /// `x * 0` becomes the literal zero.
    #[test]
    fn algebraic_simplification_annihilates_multiply_by_zero() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("mul_zero");

        let x = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let zero = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(ConstantValue::scalar(0.0)),
            vec![],
            scalar_shape(),
        );
        let product = add_op(
            &mut builder,
            &mut comp,
            OperationType::Multiply,
            vec![x, zero],
            scalar_shape(),
        );
        builder
            .set_outputs(&mut comp, &[product])
            .expect("outputs must be settable");

        let mut pass: AlgebraicSimplificationPass<f32> = AlgebraicSimplificationPass::new();
        assert!(pass.run(&mut comp).expect("simplification must succeed"));

        match comp.producer_of(product).map(|op| &op.op_type) {
            Some(OperationType::Constant(value)) => assert_eq!(value.as_scalar(), Some(0.0)),
            other => panic!("expected literal zero, got {other:?}"),
        }
    }

    /// A rewrite must not fire when it would elide an implicit broadcast.
    #[test]
    fn algebraic_simplification_respects_shapes() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("broadcast_guard");

        // Scalar x times a [4] tensor of ones broadcasts to [4]; forwarding the
        // scalar would change the result shape.
        let x = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let ones = add_op(
            &mut builder,
            &mut comp,
            OperationType::Constant(
                ConstantValue::new(vec![1.0; 4], vec![4]).expect("valid constant"),
            ),
            vec![],
            shape(&[4]),
        );
        let product = add_op(
            &mut builder,
            &mut comp,
            OperationType::Multiply,
            vec![x, ones],
            shape(&[4]),
        );
        builder
            .set_outputs(&mut comp, &[product])
            .expect("outputs must be settable");

        let mut pass: AlgebraicSimplificationPass<f32> = AlgebraicSimplificationPass::new();
        let changed = pass.run(&mut comp).expect("simplification must succeed");

        assert!(!changed, "shape-changing forwarding must be refused");
        assert!(matches!(
            comp.producer_of(product).map(|op| &op.op_type),
            Some(OperationType::Multiply)
        ));
    }

    /// F1 end-to-end: a real graph must survive the whole pipeline.
    #[test]
    fn pipeline_preserves_a_live_three_operation_graph() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("pipeline");

        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let b = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let sum = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![a, b],
            scalar_shape(),
        );
        builder
            .mark_terminal_operands_as_outputs(&mut comp)
            .expect("outputs must be inferable");
        builder
            .validate_computation(&comp)
            .expect("the graph must validate");

        let config = crate::xla::XLACompilerConfig::default();
        let mut pipeline: OptimizationPipeline<f32> = OptimizationPipeline::new(&config);
        let optimized = pipeline.optimize(comp).expect("pipeline must succeed");

        assert_eq!(
            optimized.operations.len(),
            3,
            "no live operation may be dropped by the pipeline"
        );
        assert!(optimized.producer_of(sum).is_some());
        assert_eq!(optimized.outputs.len(), 1);
    }

    /// F19: a pipeline that changed nothing must report no applied passes.
    #[test]
    fn pipeline_does_not_report_passes_that_did_nothing() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("nothing_to_do");

        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let b = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let _ = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![a, b],
            scalar_shape(),
        );
        builder
            .mark_terminal_operands_as_outputs(&mut comp)
            .expect("outputs must be inferable");

        let mut optimizer: GraphOptimizer<f32> = GraphOptimizer::new(&pipeline_config());
        let (_optimized, changed) = optimizer
            .optimize_tracked(comp)
            .expect("optimization must succeed");

        assert!(
            !changed,
            "a graph with nothing to optimize must not report a change"
        );
    }

    /// Benefit estimates must be zero when there is no opportunity.
    #[test]
    fn benefit_estimates_are_zero_without_opportunities() {
        let mut builder: ComputationGraphBuilder<f32> = ComputationGraphBuilder::new();
        let mut comp = builder.create_computation("no_benefit");

        let a = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let b = add_op(
            &mut builder,
            &mut comp,
            OperationType::Parameter,
            vec![],
            scalar_shape(),
        );
        let sum = add_op(
            &mut builder,
            &mut comp,
            OperationType::Add,
            vec![a, b],
            scalar_shape(),
        );
        builder
            .set_outputs(&mut comp, &[sum])
            .expect("outputs must be settable");

        let folder: ConstantFoldingPass<f32> = ConstantFoldingPass::new(&pipeline_config());
        let cse: CommonSubexpressionEliminationPass<f32> =
            CommonSubexpressionEliminationPass::new();
        let algebraic: AlgebraicSimplificationPass<f32> = AlgebraicSimplificationPass::new();
        let dce: DeadCodeEliminationPass<f32> = DeadCodeEliminationPass::new(&pipeline_config());

        assert_eq!(folder.estimate_benefit(&comp), 0.0);
        assert_eq!(cse.estimate_benefit(&comp), 0.0);
        assert_eq!(algebraic.estimate_benefit(&comp), 0.0);
        assert_eq!(dce.estimate_benefit(&comp), 0.0);
    }
}
