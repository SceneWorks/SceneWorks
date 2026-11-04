//! Fixture budgets represent the raw probe, so use the production ladder reserve directly.
pub(crate) fn reserve_for(budget: Option<super::VramBudget>) -> f64 {
    budget.map_or(0.0, super::ladder_reserve_gb)
}
