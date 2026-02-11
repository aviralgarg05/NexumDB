use super::types::{Column, DataType, SelectItem, Statement, Value};
use anyhow::{anyhow, Result};
use sqlparser::ast::{self, ColumnDef, DataType as SqlDataType, Expr, Statement as SqlStatement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser as SqlParser;

pub struct Parser;

impl Parser {
    pub fn parse(sql: &str) -> Result<Statement> {
        let trimmed = sql.trim().trim_end_matches(';').trim();
        if trimmed.is_empty() {
            return Err(anyhow!("No statements found"));
        }

        if let Some(statement) = Self::parse_management_statement(trimmed)? {
            return Ok(statement);
        }

        let dialect = GenericDialect {};
        let statements = SqlParser::parse_sql(&dialect, trimmed)?;

        if statements.is_empty() {
            return Err(anyhow!("No statements found"));
        }

        let stmt = &statements[0];
        Self::convert_statement(stmt)
    }

    fn convert_statement(stmt: &SqlStatement) -> Result<Statement> {
        match stmt {
            SqlStatement::CreateTable { name, columns, .. } => {
                let table_name = name.to_string();
                let cols = columns
                    .iter()
                    .map(Self::convert_column)
                    .collect::<Result<Vec<_>>>()?;
                Ok(Statement::CreateTable {
                    name: table_name,
                    columns: cols,
                })
            }
            SqlStatement::Insert {
                table_name,
                columns,
                source,
                ..
            } => {
                let table = table_name.to_string();
                let col_names = columns.iter().map(|c| c.to_string()).collect();

                let values = if let ast::SetExpr::Values(values) = &*source.body {
                    values
                        .rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .map(Self::convert_expr)
                                .collect::<Result<Vec<_>>>()
                        })
                        .collect::<Result<Vec<_>>>()?
                } else {
                    return Err(anyhow!("Unsupported INSERT format"));
                };

                Ok(Statement::Insert {
                    table,
                    columns: col_names,
                    values,
                })
            }
            SqlStatement::Update {
                table,
                assignments,
                selection,
                ..
            } => {
                let table_name = table.to_string();

                let assignment_pairs = assignments
                    .iter()
                    .map(|assign| {
                        let col_name = assign
                            .id
                            .iter()
                            .map(|i| i.value.clone())
                            .collect::<Vec<_>>()
                            .join(".");
                        let value = Self::convert_expr(&assign.value)?;
                        Ok((col_name, value))
                    })
                    .collect::<Result<Vec<_>>>()?;

                let where_clause = selection.as_ref().map(|expr| Box::new(expr.clone()));

                Ok(Statement::Update {
                    table: table_name,
                    assignments: assignment_pairs,
                    where_clause,
                })
            }
            SqlStatement::Delete {
                from, selection, ..
            } => {
                let table = if let Some(from_clause) = from.first() {
                    from_clause.relation.to_string()
                } else {
                    return Err(anyhow!("DELETE requires a table name"));
                };

                let where_clause = selection.as_ref().map(|expr| Box::new(expr.clone()));

                Ok(Statement::Delete {
                    table,
                    where_clause,
                })
            }
            SqlStatement::Query(query) => {
                if let ast::SetExpr::Select(select) = &*query.body {
                    let table =
                        if let Some(ast::TableWithJoins { relation, .. }) = select.from.first() {
                            if let ast::TableFactor::Table { name, .. } = relation {
                                name.to_string()
                            } else {
                                return Err(anyhow!("Unsupported table reference"));
                            }
                        } else {
                            return Err(anyhow!("No table specified"));
                        };

                    let projection = select
                        .projection
                        .iter()
                        .map(|proj| match proj {
                            ast::SelectItem::Wildcard(_) => Ok(SelectItem::Wildcard),
                            ast::SelectItem::UnnamedExpr(expr) => match expr {
                                Expr::Identifier(ident) => Ok(SelectItem::Column {
                                    name: ident.value.clone(),
                                    alias: None,
                                }),
                                _ => Err(anyhow!("Unsupported select expression: {}", expr)),
                            },
                            ast::SelectItem::ExprWithAlias { expr, alias } => match expr {
                                Expr::Identifier(ident) => Ok(SelectItem::Column {
                                    name: ident.value.clone(),
                                    alias: Some(alias.value.clone()),
                                }),
                                _ => Err(anyhow!("Unsupported select expression: {}", expr)),
                            },
                            _ => Err(anyhow!("Unsupported select item")),
                        })
                        .collect::<Result<Vec<_>>>()?;

                    let where_clause = select.selection.as_ref().map(|expr| Box::new(expr.clone()));

                    let order_by = if !query.order_by.is_empty() {
                        Some(
                            query
                                .order_by
                                .iter()
                                .map(|order| {
                                    let column = match &order.expr {
                                        Expr::Identifier(ident) => ident.value.clone(),
                                        _ => {
                                            return Err(anyhow!(
                                                "Unsupported ORDER BY expression: {}",
                                                order.expr
                                            ))
                                        }
                                    };
                                    let ascending = order.asc.unwrap_or(true);
                                    Ok(crate::sql::types::OrderByClause { column, ascending })
                                })
                                .collect::<Result<Vec<_>>>()?,
                        )
                    } else {
                        None
                    };

                    let limit = query.limit.as_ref().and_then(|limit_expr| {
                        if let ast::Expr::Value(ast::Value::Number(n, _)) = limit_expr {
                            n.parse().ok()
                        } else {
                            None
                        }
                    });

                    Ok(Statement::Select {
                        table,
                        projection,
                        where_clause,
                        order_by,
                        limit,
                    })
                } else {
                    Err(anyhow!("Unsupported query type"))
                }
            }

            SqlStatement::Drop {
                object_type,
                if_exists,
                names,
                ..
            } => {
                // Handle DROP TABLE statements parsed by sqlparser
                match object_type {
                    ast::ObjectType::Table => {
                        if names.len() != 1 {
                            return Err(anyhow!("Can only drop one table at a time"));
                        }
                        let table_name = names[0].to_string();
                        Ok(Statement::DropTable {
                            name: table_name,
                            if_exists: *if_exists,
                        })
                    }
                    _ => Err(anyhow!("Only DROP TABLE is supported")),
                }
            }
            _ => Err(anyhow!("Unsupported statement type")),
        }
    }

    fn parse_management_statement(sql: &str) -> Result<Option<Statement>> {
        let tokens: Vec<&str> = sql.split_whitespace().collect();
        if tokens.is_empty() {
            return Ok(None);
        }

        if tokens.len() == 2
            && tokens[0].eq_ignore_ascii_case("show")
            && tokens[1].eq_ignore_ascii_case("tables")
        {
            return Ok(Some(Statement::ShowTables));
        }

        if tokens.len() == 2 && tokens[0].eq_ignore_ascii_case("describe") {
            let table = Self::clean_identifier(tokens[1]);
            return Ok(Some(Statement::DescribeTable { name: table }));
        }

        if tokens.len() == 3
            && tokens[0].eq_ignore_ascii_case("drop")
            && tokens[1].eq_ignore_ascii_case("table")
        {
            let table = Self::clean_identifier(tokens[2]);
            return Ok(Some(Statement::DropTable {
                name: table,
                if_exists: false,
            }));
        }

        if tokens.len() == 5
            && tokens[0].eq_ignore_ascii_case("drop")
            && tokens[1].eq_ignore_ascii_case("table")
            && tokens[2].eq_ignore_ascii_case("if")
            && tokens[3].eq_ignore_ascii_case("exists")
        {
            let table = Self::clean_identifier(tokens[4]);
            return Ok(Some(Statement::DropTable {
                name: table,
                if_exists: true,
            }));
        }

        Ok(None)
    }

    fn clean_identifier(raw: &str) -> String {
        let trimmed = raw.trim();
        if trimmed.len() >= 2 {
            let first = trimmed.chars().next().unwrap();
            let last = trimmed.chars().last().unwrap();
            if (first == '`' && last == '`')
                || (first == '"' && last == '"')
                || (first == '\'' && last == '\'')
            {
                return trimmed[1..trimmed.len() - 1].to_string();
            }
        }
        trimmed.to_string()
    }

    fn convert_column(col: &ColumnDef) -> Result<Column> {
        let name = col.name.to_string();
        let data_type = Self::convert_data_type(&col.data_type)?;
        Ok(Column { name, data_type })
    }

    fn convert_data_type(data_type: &SqlDataType) -> Result<DataType> {
        match data_type {
            SqlDataType::Int(_) | SqlDataType::Integer(_) | SqlDataType::BigInt(_) => {
                Ok(DataType::Integer)
            }
            SqlDataType::Float(_) | SqlDataType::Double | SqlDataType::Real => Ok(DataType::Float),
            SqlDataType::Text
            | SqlDataType::Varchar(_)
            | SqlDataType::Char(_)
            | SqlDataType::String(_) => Ok(DataType::Text),
            SqlDataType::Boolean => Ok(DataType::Boolean),
            _ => Err(anyhow!("Unsupported data type: {:?}", data_type)),
        }
    }

    fn convert_expr(expr: &Expr) -> Result<Value> {
        match expr {
            Expr::Value(ast::Value::Number(n, _)) => {
                if n.contains('.') {
                    Ok(Value::Float(n.parse()?))
                } else {
                    Ok(Value::Integer(n.parse()?))
                }
            }
            Expr::Value(ast::Value::SingleQuotedString(s))
            | Expr::Value(ast::Value::DoubleQuotedString(s)) => Ok(Value::Text(s.clone())),
            Expr::Value(ast::Value::Boolean(b)) => Ok(Value::Boolean(*b)),
            Expr::Value(ast::Value::Null) => Ok(Value::Null),
            _ => Err(anyhow!("Unsupported expression: {:?}", expr)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_create_table() {
        let sql = "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::CreateTable { name, columns } => {
                assert_eq!(name, "users");
                assert_eq!(columns.len(), 3);
                assert_eq!(columns[0].name, "id");
                assert_eq!(columns[1].name, "name");
            }
            _ => panic!("Expected CreateTable statement"),
        }
    }

    #[test]
    fn test_parse_insert() {
        let sql = "INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Insert {
                table,
                columns,
                values,
            } => {
                assert_eq!(table, "users");
                assert_eq!(columns.len(), 2);
                assert_eq!(values.len(), 2);
            }
            _ => panic!("Expected Insert statement"),
        }
    }

    #[test]
    fn test_parse_select() {
        let sql = "SELECT id, name FROM users";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Select {
                table, projection, ..
            } => {
                assert_eq!(table, "users");
                assert_eq!(projection.len(), 2);
            }
            _ => panic!("Expected Select statement"),
        }
    }

    #[test]
    fn test_parse_select_wildcard() {
        let sql = "SELECT * FROM users";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Select { projection, .. } => {
                assert_eq!(projection.len(), 1);
                match &projection[0] {
                    SelectItem::Wildcard => {}
                    _ => panic!("Expected wildcard projection"),
                }
            }
            _ => panic!("Expected Select statement"),
        }
    }

    #[test]
    fn test_parse_select_alias() {
        let sql = "SELECT name AS username FROM users";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Select { projection, .. } => {
                assert_eq!(projection.len(), 1);
                match &projection[0] {
                    SelectItem::Column { name, alias } => {
                        assert_eq!(name, "name");
                        assert_eq!(alias.as_deref(), Some("username"));
                    }
                    _ => panic!("Expected column projection"),
                }
            }
            _ => panic!("Expected Select statement"),
        }
    }

    #[test]
    fn test_parse_update_single_column() {
        let sql = "UPDATE users SET name = 'Bob' WHERE id = 1";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Update {
                table,
                assignments,
                where_clause,
            } => {
                assert_eq!(table, "users");
                assert_eq!(assignments.len(), 1);
                assert_eq!(assignments[0].0, "name");
                assert_eq!(assignments[0].1, Value::Text("Bob".to_string()));
                assert!(where_clause.is_some());
            }
            _ => panic!("Expected Update statement"),
        }
    }

    #[test]
    fn test_parse_delete_with_where() {
        let sql = "DELETE FROM users WHERE id = 1";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Delete {
                table,
                where_clause,
            } => {
                assert_eq!(table, "users");
                assert!(where_clause.is_some());
            }
            _ => panic!("Expected Delete statement"),
        }
    }

    #[test]
    fn test_parse_update_multiple_columns() {
        let sql = "UPDATE users SET name = 'Bob', age = 30 WHERE id = 1";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Update {
                table,
                assignments,
                where_clause,
            } => {
                assert_eq!(table, "users");
                assert_eq!(assignments.len(), 2);
                assert_eq!(assignments[0].0, "name");
                assert_eq!(assignments[0].1, Value::Text("Bob".to_string()));
                assert_eq!(assignments[1].0, "age");
                assert_eq!(assignments[1].1, Value::Integer(30));
                assert!(where_clause.is_some());
            }
            _ => panic!("Expected Update statement"),
        }
    }

    #[test]
    fn test_parse_update_without_where() {
        let sql = "UPDATE users SET active = true";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Update {
                table,
                assignments,
                where_clause,
            } => {
                assert_eq!(table, "users");
                assert_eq!(assignments.len(), 1);
                assert_eq!(assignments[0].0, "active");
                assert_eq!(assignments[0].1, Value::Boolean(true));
                assert!(where_clause.is_none());
            }
            _ => panic!("Expected Update statement"),
        }
    }

    #[test]
    fn test_parse_delete_without_where() {
        let sql = "DELETE FROM users";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::Delete {
                table,
                where_clause,
            } => {
                assert_eq!(table, "users");
                assert!(where_clause.is_none());
            }
            _ => panic!("Expected Delete statement"),
        }
    }

    #[test]
    fn test_parse_show_tables() {
        let sql = "SHOW TABLES";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::ShowTables => {}
            _ => panic!("Expected ShowTables statement"),
        }
    }

    #[test]
    fn test_parse_describe_table() {
        let sql = "DESCRIBE users";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::DescribeTable { name } => {
                assert_eq!(name, "users");
            }
            _ => panic!("Expected DescribeTable statement"),
        }
    }

    #[test]
    fn test_parse_drop_table() {
        let sql = "DROP TABLE users";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::DropTable { name, if_exists } => {
                assert_eq!(name, "users");
                assert!(!if_exists);
            }
            _ => panic!("Expected DropTable statement"),
        }
    }

    #[test]
    fn test_parse_drop_table_if_exists() {
        let sql = "DROP TABLE IF EXISTS users";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::DropTable { name, if_exists } => {
                assert_eq!(name, "users");
                assert!(if_exists);
            }
            _ => panic!("Expected DropTable statement"),
        }
    }

    #[test]
    fn test_parse_drop_table_with_quotes() {
        let sql = "DROP TABLE `users`";
        let stmt = Parser::parse(sql).unwrap();

        match stmt {
            Statement::DropTable { name, if_exists } => {
                assert_eq!(name, "users");
                assert!(!if_exists);
            }
            _ => panic!("Expected DropTable statement"),
        }
    }

    #[test]
    fn test_parse_drop_table_sqlparser_syntax() {
        // Test that sqlparser's DROP TABLE syntax also works
        let sql = "DROP TABLE users";
        let stmt = Parser::parse(sql).unwrap();
        
        match stmt {
            Statement::DropTable { name, if_exists } => {
                assert_eq!(name, "users");
                assert!(!if_exists);
            }
            _ => panic!("Expected DropTable statement"),
        }
    }
}
