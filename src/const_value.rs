use std::{ops::{Add, Sub, Mul, Div}, sync::Arc};

#[derive(Clone, Debug, PartialEq)]
pub enum ConstValue {
    Int { value: i64 },
    UInt { value: u64 },
    Float32 { value: f32 },
    String { value: Arc<str> },
    Char { value: char },
    Bool { value: bool },
}

impl ConstValue {
    pub fn apply_numeric_to_numeric(self, other: ConstValue, operation_int: fn(i64, i64) -> i64, operation_uint: fn(u64, u64) -> u64, operation_float: fn(f32, f32) -> f32) -> Option<ConstValue> {
        match self {
            ConstValue::Int { value } => {
                let ConstValue::Int { value: other_value } = other else {
                    return None;
                };

                Some(ConstValue::Int {
                    value: operation_int(value, other_value),
                })
            }
            ConstValue::UInt { value } => {
                let ConstValue::UInt { value: other_value } = other else {
                    return None;
                };

                Some(ConstValue::UInt {
                    value: operation_uint(value, other_value),
                })
            }
            ConstValue::Float32 { value } => {
                let ConstValue::Float32 { value: other_value } = other else {
                    return None;
                };

                Some(ConstValue::Float32 {
                    value: operation_float(value, other_value),
                })
            }
            _ => None,
        }
    }

    pub fn apply_numeric_to_bool(self, other: ConstValue, operation_int: fn(&i64, &i64) -> bool, operation_uint: fn(&u64, &u64) -> bool, operation_float: fn(&f32, &f32) -> bool) -> Option<ConstValue> {
        match self {
            ConstValue::Int { value } => {
                let ConstValue::Int { value: other_value } = other else {
                    return None;
                };

                Some(ConstValue::Bool {
                    value: operation_int(&value, &other_value),
                })
            }
            ConstValue::UInt { value } => {
                let ConstValue::UInt { value: other_value } = other else {
                    return None;
                };

                Some(ConstValue::Bool {
                    value: operation_uint(&value, &other_value),
                })
            }
            ConstValue::Float32 { value } => {
                let ConstValue::Float32 { value: other_value } = other else {
                    return None;
                };

                Some(ConstValue::Bool {
                    value: operation_float(&value, &other_value),
                })
            }
            _ => None,
        }
    }

    pub fn add(self, other: ConstValue) -> Option<ConstValue> {
        self.apply_numeric_to_numeric(other, i64::add, u64::add, f32::add)
    }

    pub fn subtract(self, other: ConstValue) -> Option<ConstValue> {
        self.apply_numeric_to_numeric(other, i64::sub, u64::sub, f32::sub)
    }

    pub fn multiply(self, other: ConstValue) -> Option<ConstValue> {
        self.apply_numeric_to_numeric(other, i64::mul, u64::mul, f32::mul)
    }

    pub fn divide(self, other: ConstValue) -> Option<ConstValue> {
        self.apply_numeric_to_numeric(other, i64::div, u64::div, f32::div)
    }

    pub fn less(self, other: ConstValue) -> Option<ConstValue> {
        self.apply_numeric_to_bool(other, i64::lt, u64::lt, f32::lt)
    }

    pub fn greater(self, other: ConstValue) -> Option<ConstValue> {
        self.apply_numeric_to_bool(other, i64::gt, u64::gt, f32::gt)
    }

    pub fn less_equal(self, other: ConstValue) -> Option<ConstValue> {
        self.apply_numeric_to_bool(other, i64::le, u64::le, f32::le)
    }

    pub fn greater_equal(self, other: ConstValue) -> Option<ConstValue> {
        self.apply_numeric_to_bool(other, i64::ge, u64::ge, f32::ge)
    }

    pub fn and(self, other: ConstValue) -> Option<ConstValue> {
        let ConstValue::Bool { value } = self else {
            return None;
        };

        let ConstValue::Bool { value: other_value } = other else {
            return None;
        };

        Some(ConstValue::Bool {
            value: value && other_value
        })
    }

    pub fn or(self, other: ConstValue) -> Option<ConstValue> {
        let ConstValue::Bool { value } = self else {
            return None;
        };

        let ConstValue::Bool { value: other_value } = other else {
            return None;
        };

        Some(ConstValue::Bool {
            value: value || other_value
        })
    }
}
