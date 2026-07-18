package com.relativedb.rt;

/** Failure in the native RT backend (library loading, checkpoint loading, forward pass). */
public class RtException extends RuntimeException {
    public RtException(String message) { super(message); }
    public RtException(String message, Throwable cause) { super(message, cause); }
}
