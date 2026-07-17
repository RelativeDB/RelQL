//! Schema declaration: tables, columns, links, value types.
//!
//! Only *shape* lives here — no URLs, no credentials, no connectors.
//! Mirrors `dev.relativedb.schema` (Java) / `relativedb.schema` (Python).

use std::collections::HashMap;
use std::fmt;

/// Semantic value types — exactly RT's sem types (F10–F13).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ValueType {
    Number,
    Text,
    Datetime,
    Boolean,
}

/// Raised when a schema is internally inconsistent.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SchemaError(pub String);

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "schema error: {}", self.0)
    }
}
impl std::error::Error for SchemaError {}

/// A typed feature column. IDs / FK columns are *not* columns here (F17).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ColumnDef {
    pub name: String,
    pub value_type: ValueType,
}

impl ColumnDef {
    pub fn of(name: impl Into<String>, value_type: ValueType) -> Self {
        ColumnDef { name: name.into(), value_type }
    }
}

/// A foreign-key link: `from_table.fk_column -> to_table.primary_key`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct LinkDef {
    pub from_table: String,
    pub fk_column: String,
    pub to_table: String,
}

impl LinkDef {
    pub fn link(
        from_table: impl Into<String>,
        fk_column: impl Into<String>,
        to_table: impl Into<String>,
    ) -> Self {
        LinkDef {
            from_table: from_table.into(),
            fk_column: fk_column.into(),
            to_table: to_table.into(),
        }
    }
}

/// A table: typed feature columns + identity (PK) + optional row time.
///
/// The primary key is identity only — never surfaced as a cell (F17).
/// `time_column` drives temporal filtering (F24) and windows.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TableDef {
    pub name: String,
    pub columns: Vec<ColumnDef>,
    pub primary_key: Option<String>,
    pub time_column: Option<String>,
}

impl TableDef {
    pub fn new_table(name: impl Into<String>) -> TableBuilder {
        TableBuilder {
            name: name.into(),
            columns: Vec::new(),
            primary_key: None,
            time_column: None,
        }
    }

    pub fn column(&self, name: &str) -> Option<&ColumnDef> {
        self.columns.iter().find(|c| c.name == name)
    }
}

/// Builder for [`TableDef`]. `build` validates duplicate columns and that the
/// declared `time_column` is a real column (mirrors Python `__post_init__`).
pub struct TableBuilder {
    name: String,
    columns: Vec<ColumnDef>,
    primary_key: Option<String>,
    time_column: Option<String>,
}

impl TableBuilder {
    pub fn column(mut self, name: impl Into<String>, value_type: ValueType) -> Self {
        self.columns.push(ColumnDef::of(name, value_type));
        self
    }

    pub fn column_def(mut self, column: ColumnDef) -> Self {
        self.columns.push(column);
        self
    }

    pub fn primary_key(mut self, column: impl Into<String>) -> Self {
        self.primary_key = Some(column.into());
        self
    }

    pub fn time_column(mut self, column: impl Into<String>) -> Self {
        self.time_column = Some(column.into());
        self
    }

    /// Fallible construction: duplicate columns / undeclared `time_column` fail.
    pub fn try_build(self) -> Result<TableDef, SchemaError> {
        let mut seen = std::collections::HashSet::new();
        for c in &self.columns {
            if !seen.insert(c.name.clone()) {
                return Err(SchemaError(format!(
                    "table {:?}: duplicate column {:?}",
                    self.name, c.name
                )));
            }
        }
        if let Some(tc) = &self.time_column {
            if !seen.contains(tc) {
                return Err(SchemaError(format!(
                    "table {:?}: time_column {:?} is not a declared column",
                    self.name, tc
                )));
            }
        }
        Ok(TableDef {
            name: self.name,
            columns: self.columns,
            primary_key: self.primary_key,
            time_column: self.time_column,
        })
    }

    /// Convenience: panics on an inconsistent table (matches the ergonomic
    /// builder style of the Java/Python peers).
    pub fn build(self) -> TableDef {
        self.try_build().expect("invalid TableDef")
    }
}

/// The declared relational graph. Validates on construction.
#[derive(Clone, Debug)]
pub struct Schema {
    pub tables: Vec<TableDef>,
    pub links: Vec<LinkDef>,
    by_name: HashMap<String, usize>,
}

impl Schema {
    pub fn new_schema() -> SchemaBuilder {
        SchemaBuilder { tables: Vec::new(), links: Vec::new() }
    }

    pub fn table(&self, name: &str) -> Option<&TableDef> {
        self.by_name.get(name).map(|&i| &self.tables[i])
    }

    pub fn require_table(&self, name: &str) -> Result<&TableDef, SchemaError> {
        self.table(name)
            .ok_or_else(|| SchemaError(format!("unknown table {:?}", name)))
    }

    /// F→P links whose *from* side is `table` (its parents).
    pub fn links_from(&self, table: &str) -> Vec<&LinkDef> {
        self.links.iter().filter(|l| l.from_table == table).collect()
    }

    /// P→F links whose *to* side is `table` (its children edges).
    pub fn links_to(&self, table: &str) -> Vec<&LinkDef> {
        self.links.iter().filter(|l| l.to_table == table).collect()
    }
}

pub struct SchemaBuilder {
    tables: Vec<TableDef>,
    links: Vec<LinkDef>,
}

impl SchemaBuilder {
    pub fn table(mut self, table: TableDef) -> Self {
        self.tables.push(table);
        self
    }

    pub fn link(mut self, link: LinkDef) -> Self {
        self.links.push(link);
        self
    }

    pub fn try_build(self) -> Result<Schema, SchemaError> {
        let mut by_name = HashMap::new();
        for (i, t) in self.tables.iter().enumerate() {
            if by_name.insert(t.name.clone(), i).is_some() {
                return Err(SchemaError(format!("duplicate table {:?}", t.name)));
            }
        }
        for l in &self.links {
            let from = by_name
                .get(&l.from_table)
                .map(|&i| &self.tables[i])
                .ok_or_else(|| {
                    SchemaError(format!("link {:?}: unknown from_table {:?}", l, l.from_table))
                })?;
            let _ = from;
            let to = by_name
                .get(&l.to_table)
                .map(|&i| &self.tables[i])
                .ok_or_else(|| {
                    SchemaError(format!("link {:?}: unknown to_table {:?}", l, l.to_table))
                })?;
            // Link targets need PKs: the F→P edge resolves to the parent's PK.
            if to.primary_key.is_none() {
                return Err(SchemaError(format!(
                    "link {:?}: to_table {:?} has no primary key",
                    l, l.to_table
                )));
            }
        }
        // The F17 invariant: PK/FK columns may not be declared feature columns.
        let fk_cols: std::collections::HashSet<(&str, &str)> = self
            .links
            .iter()
            .map(|l| (l.from_table.as_str(), l.fk_column.as_str()))
            .collect();
        for t in &self.tables {
            if let Some(pk) = &t.primary_key {
                if t.columns.iter().any(|c| &c.name == pk) {
                    return Err(SchemaError(format!(
                        "table {:?}: primary key {:?} may not also be a feature column (F17)",
                        t.name, pk
                    )));
                }
            }
            for c in &t.columns {
                if fk_cols.contains(&(t.name.as_str(), c.name.as_str())) {
                    return Err(SchemaError(format!(
                        "table {:?}: FK column {:?} may not also be a feature column (F17)",
                        t.name, c.name
                    )));
                }
            }
        }
        Ok(Schema { tables: self.tables, links: self.links, by_name })
    }

    pub fn build(self) -> Schema {
        self.try_build().expect("invalid Schema")
    }
}
