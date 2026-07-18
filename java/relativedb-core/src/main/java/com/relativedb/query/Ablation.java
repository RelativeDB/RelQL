package com.relativedb.query;

/** An {@code ABLATE TABLE <name>} directive. {@code kind} is currently always {@code "table"}. */
public record Ablation(String kind, String name) { }
