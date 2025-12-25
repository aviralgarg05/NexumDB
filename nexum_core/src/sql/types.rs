use serde::{Deserialize, Serialize};
use sqlparser::ast::Expr;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    Integer,
    Float,
    Text,
    Boolean,
    Null,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Text(String),
    Boolean(bool),
    Null,
}

impl Value {
    pub fn data_type(&self) -> DataType {
        match self {
            Value::Integer(_) => DataType::Integer,
            Value::Float(_) => DataType::Float,
            Value::Text(_) => DataType::Text,
            Value::Boolean(_) => DataType::Boolean,
            Value::Null => DataType::Null,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub data_type: DataType,
}

#[derive(Debug, Clone)]
pub struct TableSchema {
    pub name: String,
    pub columns: Vec<Column>,
}

#[derive(Debug, Clone)]
pub struct OrderByClause {
    pub column: String,
    pub ascending: bool,
}

#[derive(Debug, Clone)]
pub enum Statement {
    CreateTable {
        name: String,
        columns: Vec<Column>,
    },
    Insert {
        table: String,
        columns: Vec<String>,
        values: Vec<Vec<Value>>,
    },
    Select {
        table: String,
        columns: Vec<String>,
        where_clause: Option<Box<Expr>>,
        order_by: Option<Vec<OrderByClause>>,
        limit: Option<usize>,
    },
    Delete {
        table: String,
        where_clause: Option<Box<Expr>>,
    },
    Update {
        table: String,
        assignments: Vec<(String, Value)>,
        where_clause: Option<Box<Expr>>,
    },
}
