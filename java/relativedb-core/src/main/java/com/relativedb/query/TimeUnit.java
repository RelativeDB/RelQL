package com.relativedb.query;

import java.time.Duration;

/** PQL window time units. */
public enum TimeUnit {
    SECONDS(Duration.ofSeconds(1)),
    MINUTES(Duration.ofMinutes(1)),
    HOURS(Duration.ofHours(1)),
    DAYS(Duration.ofDays(1)),
    WEEKS(Duration.ofDays(7)),
    MONTHS(Duration.ofDays(30));   // calendar-free approximation

    private final Duration unit;
    TimeUnit(Duration unit) { this.unit = unit; }
    public Duration duration() { return unit; }
}
