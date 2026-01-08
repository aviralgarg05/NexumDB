//! SQL expression evaluation and filtering module.
//!
//! This module provides functionality for evaluating SQL WHERE clause expressions
//! against row data. It supports a wide range of SQL operations including:
//!
//! - Comparison operators (=, !=, <, >, <=, >=)
//! - Logical operators (AND, OR)
//! - Pattern matching (LIKE with % and _ wildcards)
//! - List membership (IN, NOT IN)
//! - Range queries (BETWEEN, NOT BETWEEN)
//!
//! # Primary Types
//!
//! - [`ExpressionEvaluator`]: The main evaluator that processes SQL expressions
//!   against row data using column names for field resolution.
//!
//! # Usage
//!
//! ```rust
//! use nexum_core::executor::filter::ExpressionEvaluator;
//! use nexum_core::sql::types::Value;
//! use sqlparser::ast::Expr;
//!
//! // Create evaluator with column schema
//! let columns = vec!["id".to_string(), "name".to_string(), "age".to_string()];
//! let evaluator = ExpressionEvaluator::new(columns);
//!
//! // Evaluate expression against row data
//! let row_data = vec![
//!     Value::Integer(1),
//!     Value::Text("Alice".to_string()),
//!     Value::Integer(30),
//! ];
//!
//! // The expression would be parsed from SQL: "age > 25 AND name = 'Alice'"
//! // let result = evaluator.evaluate(&parsed_expr, &row_data)?;
//! ```
//!
//! This module is used internally by the query executor to filter rows
//! based on WHERE clause conditions, but is made public to support
//! benchmarking and testing of filter performance.

use crate::sql::types::Value;
use anyhow::{anyhow, Result};
use regex::Regex;
use sqlparser::ast::{BinaryOperator, Expr, Value as SqlValue};

pub struct ExpressionEvaluator {
    column_names: Vec<String>,
}

impl ExpressionEvaluator {
    pub fn new(column_names: Vec<String>) -> Self {
        Self { column_names }
    }

    pub fn evaluate(&self, expr: &Expr, row_values: &[Value]) -> Result<bool> {
        match expr {
            Expr::BinaryOp { left, op, right } => {
                self.evaluate_binary_op(left, op, right, row_values)
            }
            Expr::Identifier(ident) => {
                let col_name = ident.value.as_str();
                let idx = self
                    .column_names
                    .iter()
                    .position(|name| name == col_name)
                    .ok_or_else(|| anyhow!("Column {} not found", col_name))?;

                match &row_values[idx] {
                    Value::Boolean(b) => Ok(*b),
                    _ => Err(anyhow!("Expected boolean value for identifier")),
                }
            }
            Expr::Like {
                negated,
                expr,
                pattern,
                escape_char,
            } => self.evaluate_like(*negated, expr, pattern, escape_char.as_ref(), row_values),
            Expr::InList {
                expr,
                list,
                negated,
            } => self.evaluate_in_list(expr, list, *negated, row_values),
            Expr::Between {
                expr,
                negated,
                low,
                high,
            } => self.evaluate_between(expr, *negated, low, high, row_values),
            _ => Err(anyhow!("Unsupported expression type: {:?}", expr)),
        }
    }

    fn evaluate_binary_op(
        &self,
        left: &Expr,
        op: &BinaryOperator,
        right: &Expr,
        row_values: &[Value],
    ) -> Result<bool> {
        match op {
            BinaryOperator::And => {
                let left_result = self.evaluate(left, row_values)?;
                let right_result = self.evaluate(right, row_values)?;
                Ok(left_result && right_result)
            }
            BinaryOperator::Or => {
                let left_result = self.evaluate(left, row_values)?;
                let right_result = self.evaluate(right, row_values)?;
                Ok(left_result || right_result)
            }
            BinaryOperator::Gt
            | BinaryOperator::Lt
            | BinaryOperator::GtEq
            | BinaryOperator::LtEq
            | BinaryOperator::Eq
            | BinaryOperator::NotEq => {
                let left_val = self.extract_value(left, row_values)?;
                let right_val = self.extract_value(right, row_values)?;
                self.compare_values(&left_val, op, &right_val)
            }
            _ => Err(anyhow!("Unsupported operator: {:?}", op)),
        }
    }

    fn extract_value(&self, expr: &Expr, row_values: &[Value]) -> Result<Value> {
        match expr {
            Expr::Identifier(ident) => {
                let col_name = ident.value.as_str();
                let idx = self
                    .column_names
                    .iter()
                    .position(|name| name == col_name)
                    .ok_or_else(|| anyhow!("Column {} not found", col_name))?;
                Ok(row_values[idx].clone())
            }
            Expr::Value(sql_val) => self.convert_sql_value(sql_val),
            _ => Err(anyhow!("Cannot extract value from expression: {:?}", expr)),
        }
    }

    fn convert_sql_value(&self, sql_val: &SqlValue) -> Result<Value> {
        match sql_val {
            SqlValue::Number(n, _) => {
                if n.contains('.') {
                    Ok(Value::Float(n.parse()?))
                } else {
                    Ok(Value::Integer(n.parse()?))
                }
            }
            SqlValue::SingleQuotedString(s) | SqlValue::DoubleQuotedString(s) => {
                Ok(Value::Text(s.clone()))
            }
            SqlValue::Boolean(b) => Ok(Value::Boolean(*b)),
            SqlValue::Null => Ok(Value::Null),
            _ => Err(anyhow!("Unsupported SQL value: {:?}", sql_val)),
        }
    }

    fn compare_values(&self, left: &Value, op: &BinaryOperator, right: &Value) -> Result<bool> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => Ok(match op {
                BinaryOperator::Eq => l == r,
                BinaryOperator::NotEq => l != r,
                BinaryOperator::Gt => l > r,
                BinaryOperator::Lt => l < r,
                BinaryOperator::GtEq => l >= r,
                BinaryOperator::LtEq => l <= r,
                _ => return Err(anyhow!("Invalid operator for integers")),
            }),
            (Value::Float(l), Value::Float(r)) => Ok(match op {
                BinaryOperator::Eq => (l - r).abs() < f64::EPSILON,
                BinaryOperator::NotEq => (l - r).abs() >= f64::EPSILON,
                BinaryOperator::Gt => l > r,
                BinaryOperator::Lt => l < r,
                BinaryOperator::GtEq => l >= r,
                BinaryOperator::LtEq => l <= r,
                _ => return Err(anyhow!("Invalid operator for floats")),
            }),
            (Value::Text(l), Value::Text(r)) => Ok(match op {
                BinaryOperator::Eq => l == r,
                BinaryOperator::NotEq => l != r,
                BinaryOperator::Gt => l > r,
                BinaryOperator::Lt => l < r,
                BinaryOperator::GtEq => l >= r,
                BinaryOperator::LtEq => l <= r,
                _ => return Err(anyhow!("Invalid operator for text")),
            }),
            (Value::Boolean(l), Value::Boolean(r)) => Ok(match op {
                BinaryOperator::Eq => l == r,
                BinaryOperator::NotEq => l != r,
                _ => return Err(anyhow!("Invalid operator for booleans")),
            }),
            (Value::Null, Value::Null) => Ok(match op {
                BinaryOperator::Eq => true,
                BinaryOperator::NotEq => false,
                _ => return Err(anyhow!("Invalid operator for nulls")),
            }),
            _ => Err(anyhow!(
                "Type mismatch in comparison: {:?} vs {:?}",
                left,
                right
            )),
        }
    }

    fn evaluate_like(
        &self,
        negated: bool,
        expr: &Expr,
        pattern: &Expr,
        _escape_char: Option<&char>,
        row_values: &[Value],
    ) -> Result<bool> {
        let value = self.extract_value(expr, row_values)?;
        let pattern_val = self.extract_value(pattern, row_values)?;

        if let (Value::Text(text), Value::Text(pat)) = (value, pattern_val) {
            let regex_pattern = pat.replace('%', ".*").replace('_', ".");

            let regex = Regex::new(&format!("^{}$", regex_pattern))
                .map_err(|e| anyhow!("Invalid LIKE pattern: {}", e))?;

            let matches = regex.is_match(&text);
            Ok(if negated { !matches } else { matches })
        } else {
            Err(anyhow!("LIKE operator requires text values"))
        }
    }

    fn evaluate_in_list(
        &self,
        expr: &Expr,
        list: &[Expr],
        negated: bool,
        row_values: &[Value],
    ) -> Result<bool> {
        let value = self.extract_value(expr, row_values)?;

        let mut found = false;
        for list_expr in list {
            let list_val = self.extract_value(list_expr, row_values)?;
            if self.values_equal(&value, &list_val) {
                found = true;
                break;
            }
        }

        Ok(if negated { !found } else { found })
    }

    fn evaluate_between(
        &self,
        expr: &Expr,
        negated: bool,
        low: &Expr,
        high: &Expr,
        row_values: &[Value],
    ) -> Result<bool> {
        let value = self.extract_value(expr, row_values)?;
        let low_val = self.extract_value(low, row_values)?;
        let high_val = self.extract_value(high, row_values)?;

        let gte_low = self.compare_values(&value, &BinaryOperator::GtEq, &low_val)?;
        let lte_high = self.compare_values(&value, &BinaryOperator::LtEq, &high_val)?;

        let in_range = gte_low && lte_high;
        Ok(if negated { !in_range } else { in_range })
    }

    fn values_equal(&self, left: &Value, right: &Value) -> bool {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => l == r,
            (Value::Float(l), Value::Float(r)) => (l - r).abs() < f64::EPSILON,
            (Value::Text(l), Value::Text(r)) => l == r,
            (Value::Boolean(l), Value::Boolean(r)) => l == r,
            (Value::Null, Value::Null) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    #[test]
    fn test_simple_comparison() {
        let column_names = vec!["id".to_string(), "name".to_string(), "age".to_string()];
        let evaluator = ExpressionEvaluator::new(column_names);

        let row_values = vec![
            Value::Integer(1),
            Value::Text("Alice".to_string()),
            Value::Integer(30),
        ];

        let sql = "age > 25";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, &format!("SELECT * FROM t WHERE {}", sql)).unwrap();

        if let sqlparser::ast::Statement::Query(query) = &ast[0] {
            if let sqlparser::ast::SetExpr::Select(select) = &*query.body {
                if let Some(where_expr) = &select.selection {
                    let result = evaluator.evaluate(where_expr, &row_values).unwrap();
                    assert!(result);
                }
            }
        }
    }

    #[test]
    fn test_and_operator() {
        let column_names = vec!["id".to_string(), "age".to_string()];
        let evaluator = ExpressionEvaluator::new(column_names);

        let row_values = vec![Value::Integer(1), Value::Integer(30)];

        let sql = "id = 1 AND age > 25";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, &format!("SELECT * FROM t WHERE {}", sql)).unwrap();

        if let sqlparser::ast::Statement::Query(query) = &ast[0] {
            if let sqlparser::ast::SetExpr::Select(select) = &*query.body {
                if let Some(where_expr) = &select.selection {
                    let result = evaluator.evaluate(where_expr, &row_values).unwrap();
                    assert!(result);
                }
            }
        }
    }

    #[test]
    fn test_text_comparison() {
        let column_names = vec!["name".to_string()];
        let evaluator = ExpressionEvaluator::new(column_names);

        let row_values = vec![Value::Text("Alice".to_string())];

        let sql = "name = 'Alice'";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, &format!("SELECT * FROM t WHERE {}", sql)).unwrap();

        if let sqlparser::ast::Statement::Query(query) = &ast[0] {
            if let sqlparser::ast::SetExpr::Select(select) = &*query.body {
                if let Some(where_expr) = &select.selection {
                    let result = evaluator.evaluate(where_expr, &row_values).unwrap();
                    assert!(result);
                }
            }
        }
    }

    #[test]
    fn test_like_operator() {
        let column_names = vec!["name".to_string()];
        let evaluator = ExpressionEvaluator::new(column_names);

        let row_values = vec![Value::Text("TestABC".to_string())];

        let sql = "name LIKE 'Test%'";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, &format!("SELECT * FROM t WHERE {}", sql)).unwrap();

        if let sqlparser::ast::Statement::Query(query) = &ast[0] {
            if let sqlparser::ast::SetExpr::Select(select) = &*query.body {
                if let Some(where_expr) = &select.selection {
                    let result = evaluator.evaluate(where_expr, &row_values).unwrap();
                    assert!(result);
                }
            }
        }
    }

    #[test]
    fn test_in_operator() {
        let column_names = vec!["status".to_string()];
        let evaluator = ExpressionEvaluator::new(column_names);

        let row_values = vec![Value::Text("active".to_string())];

        let sql = "status IN ('active', 'pending')";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, &format!("SELECT * FROM t WHERE {}", sql)).unwrap();

        if let sqlparser::ast::Statement::Query(query) = &ast[0] {
            if let sqlparser::ast::SetExpr::Select(select) = &*query.body {
                if let Some(where_expr) = &select.selection {
                    let result = evaluator.evaluate(where_expr, &row_values).unwrap();
                    assert!(result);
                }
            }
        }
    }

    #[test]
    fn test_between_operator() {
        let column_names = vec!["price".to_string()];
        let evaluator = ExpressionEvaluator::new(column_names);

        let row_values = vec![Value::Integer(150)];

        let sql = "price BETWEEN 100 AND 200";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, &format!("SELECT * FROM t WHERE {}", sql)).unwrap();

        if let sqlparser::ast::Statement::Query(query) = &ast[0] {
            if let sqlparser::ast::SetExpr::Select(select) = &*query.body {
                if let Some(where_expr) = &select.selection {
                    let result = evaluator.evaluate(where_expr, &row_values).unwrap();
                    assert!(result);
                }
            }
        }
    }
}
