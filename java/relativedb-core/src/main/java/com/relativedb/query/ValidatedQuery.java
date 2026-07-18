package com.relativedb.query;

/** A parsed query that passed schema binding, with its inferred task type. */
public record ValidatedQuery(ParsedQuery query, TaskType taskType) { }
