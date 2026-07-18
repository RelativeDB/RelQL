package com.relativedb.schema;

/** Thrown when a schema declaration is structurally invalid. */
public class SchemaException extends RuntimeException {
    public SchemaException(String message) { super(message); }
}
