package com.relativedb.query;

/** Semantic (schema-binding) error in an otherwise well-formed query. */
public class PqlValidationException extends RuntimeException {
    public PqlValidationException(String message) { super(message); }
}
