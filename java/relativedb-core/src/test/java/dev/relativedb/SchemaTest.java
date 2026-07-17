package dev.relativedb;

import dev.relativedb.schema.RelativeDbSchema;
import dev.relativedb.schema.ColumnDef;
import dev.relativedb.schema.LinkDef;
import dev.relativedb.schema.SchemaException;
import dev.relativedb.schema.TableDef;
import dev.relativedb.schema.ValueType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SchemaTest {

    @Test
    void columnDefCanonicalAndStringOverloadAgree() {
        TableDef a = TableDef.newTable("t").column(ColumnDef.of("x", ValueType.NUMBER)).build();
        TableDef b = TableDef.newTable("t").column("x", ValueType.NUMBER).build();
        assertEquals(a.columns(), b.columns());
    }

    @Test
    void linkToUnknownTableRejected() {
        SchemaException e = assertThrows(SchemaException.class, () -> RelativeDbSchema.newSchema()
                .table(TableDef.newTable("orders").primaryKey("id").build())
                .link(LinkDef.link("orders", "customer_id", "customers"))
                .build());
        assertTrue(e.getMessage().contains("customers"));
    }

    @Test
    void linkTargetNeedsPrimaryKey() {
        assertThrows(SchemaException.class, () -> RelativeDbSchema.newSchema()
                .table(TableDef.newTable("orders").primaryKey("id").build())
                .table(TableDef.newTable("customers").column("age", ValueType.NUMBER).build())
                .link(LinkDef.link("orders", "customer_id", "customers"))
                .build());
    }

    @Test
    void fkColumnMustNotBeAFeatureColumn() {
        // PK/FK values are edges, not data (F17).
        assertThrows(SchemaException.class, () -> RelativeDbSchema.newSchema()
                .table(TableDef.newTable("customers").primaryKey("customer_id").build())
                .table(TableDef.newTable("orders")
                        .column("customer_id", ValueType.NUMBER)   // FK leaked as a feature
                        .primaryKey("order_id").build())
                .link(LinkDef.link("orders", "customer_id", "customers"))
                .build());
    }

    @Test
    void primaryKeyMustNotBeAFeatureColumn() {
        assertThrows(SchemaException.class, () -> RelativeDbSchema.newSchema()
                .table(TableDef.newTable("customers")
                        .column("customer_id", ValueType.NUMBER)
                        .primaryKey("customer_id").build())
                .build());
    }

    @Test
    void duplicateTableRejected() {
        assertThrows(SchemaException.class, () -> RelativeDbSchema.newSchema()
                .table(TableDef.newTable("t").build())
                .table(TableDef.newTable("t").build()));
    }

    @Test
    void duplicateColumnRejected() {
        assertThrows(SchemaException.class, () -> TableDef.newTable("t")
                .column("x", ValueType.NUMBER)
                .column("x", ValueType.TEXT));
    }

    @Test
    void declaredTimeColumnMustBeDatetime() {
        assertThrows(SchemaException.class, () -> TableDef.newTable("t")
                .column("ts", ValueType.NUMBER)
                .timeColumn("ts")
                .build());
    }

    @Test
    void linkDirectionAccessors() {
        RelativeDbSchema schema = TestData.SCHEMA;
        assertEquals(1, schema.linksFrom("orders").size());   // F→P (parents)
        assertEquals(1, schema.linksTo("customers").size());  // P→F (children)
        assertTrue(schema.linksFrom("customers").isEmpty());
    }
}
