// Temporary stand-in for optirs_core::optimizers::Adam while optirs-core is mid-refactor.
use scirs2_core::ndarray::Array1;

pub struct Adam {
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
}

impl Adam {
    pub fn new_with_config(lr: f32, b1: f32, b2: f32, eps: f32, wd: f32) -> Self {
        Self {
            lr,
            b1,
            b2,
            eps,
            wd,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
    pub fn step_inplace(
        &mut self,
        params: &mut Array1<f32>,
        grads: &Array1<f32>,
    ) -> Result<(), String> {
        if self.m.len() != params.len() {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
            self.t = 0;
        }
        self.t += 1;
        let bc1 = 1.0 - self.b1.powi(self.t);
        let bc2 = 1.0 - self.b2.powi(self.t);
        for i in 0..params.len() {
            let p = params[i];
            let g = if self.wd > 0.0 {
                grads[i] + self.wd * p
            } else {
                grads[i]
            };
            self.m[i] = self.m[i] * self.b1 + g * (1.0 - self.b1);
            self.v[i] = self.v[i] * self.b2 + g * g * (1.0 - self.b2);
            let mh = self.m[i] / bc1;
            let vh = self.v[i] / bc2;
            params[i] = p - self.lr * mh / (vh.sqrt() + self.eps);
        }
        Ok(())
    }
}
