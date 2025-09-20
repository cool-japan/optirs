use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a funding source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingSource {
    pub id: String,
    pub name: String,
    pub agency: String,
    pub program: Option<String>,
    pub grant_number: Option<String>,
    pub amount: Option<f64>,
    pub currency: String,
    pub start_date: Option<chrono::DateTime<chrono::Utc>>,
    pub end_date: Option<chrono::DateTime<chrono::Utc>>,
}

/// Represents a grant or funding award
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grant {
    pub id: String,
    pub title: String,
    pub abstract_text: Option<String>,
    pub principal_investigator: String,
    pub co_investigators: Vec<String>,
    pub funding_source: FundingSource,
    pub status: GrantStatus,
}

/// Status of a grant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrantStatus {
    Draft,
    Submitted,
    UnderReview,
    Awarded,
    Rejected,
    Active,
    Completed,
}

/// Manager for funding and grants
#[derive(Debug, Default)]
pub struct FundingManager {
    grants: HashMap<String, Grant>,
    funding_sources: HashMap<String, FundingSource>,
}

impl FundingManager {
    pub fn new() -> Self {
        Self {
            grants: HashMap::new(),
            funding_sources: HashMap::new(),
        }
    }

    pub fn add_grant(&mut self, grant: Grant) {
        self.grants.insert(grant.id.clone(), grant);
    }

    pub fn get_grant(&self, id: &str) -> Option<&Grant> {
        self.grants.get(id)
    }

    pub fn add_funding_source(&mut self, source: FundingSource) {
        self.funding_sources.insert(source.id.clone(), source);
    }

    pub fn get_funding_source(&self, id: &str) -> Option<&FundingSource> {
        self.funding_sources.get(id)
    }

    pub fn list_active_grants(&self) -> Vec<&Grant> {
        self.grants
            .values()
            .filter(|g| matches!(g.status, GrantStatus::Active))
            .collect()
    }
}
