/*
 * Pql.g4 — ANTLR 4 grammar for Kumo Predictive Query Language (PQL)
 * ================================================================
 *
 * PQL is the SQL-like declarative language used by Kumo / KumoRFM to define a
 * predictive modeling task over a relational graph. A query names WHAT to
 * predict (the target), WHO to predict it for (the entity), and optional
 * constraints. Kumo compiles it into a training table + model.
 *
 * Canonical shape (clause order is significant):
 *
 *     PREDICT   <target>                       -- required: what to predict
 *     [FORECAST <N> TIMEFRAMES]                -- optional: multi-horizon forecast
 *     FOR [EACH] <entity_table>.<pkey>         -- required: the entity
 *     [WHERE     <condition>]                  -- optional: filter entities/facts
 *     [ASSUMING  <temporal_condition>]         -- optional: counterfactual
 *
 * Keywords are case-insensitive in the docs (PREDICT SUM / predict count both
 * appear); the lexer below is fully case-insensitive via letter fragments.
 * Table/column identifiers are treated as ordinary identifiers.
 */
grammar Pql;

// =========================================================================
//  PARSER RULES
// =========================================================================

/** A complete predictive query. `EOF` forces the whole input to parse. */
query
    : PREDICT target
      forecastClause?
      forEachClause
      whereClause?
      assumingClause?
      EOF
    ;

// -------------------------------------------------------------------------
//  Target  (the PREDICT clause payload)
// -------------------------------------------------------------------------

/**
 * The target is a boolean/value expression, optionally followed by a
 * ranking directive. A bare aggregation (`SUM(...)`) yields a regression
 * target; a comparison (`SUM(...) > 10`) yields binary classification; a
 * LIST_DISTINCT / multicategorical target must carry CLASSIFY or RANK TOP K.
 */
target
    : expr rankClause?
    ;

/** CLASSIFY (one binary head per class) | RANK TOP K (top-K retrieval). */
rankClause
    : CLASSIFY
    | RANK TOP INT
    ;

// -------------------------------------------------------------------------
//  FORECAST  (multi-timeframe forecasting, KumoRFM)
// -------------------------------------------------------------------------

/** e.g. `FORECAST 28 TIMEFRAMES` — predict N consecutive future windows. */
forecastClause
    : FORECAST INT TIMEFRAMES
    ;

// -------------------------------------------------------------------------
//  FOR EACH  (the entity)
// -------------------------------------------------------------------------

/**
 * Names the entity primary key predictions are made for. `FOR EACH` scores
 * every entity; the singular `FOR` (KumoRFM) may pin specific ids with
 * `= <id>` or `IN (<id>, ...)`.
 *   FOR EACH CUSTOMERS.CUSTOMER_ID
 *   FOR users.user_id = 42
 *   FOR users.user_id IN (42, 123)
 */
forEachClause
    : FOR EACH? columnRef entitySelector?
    ;

entitySelector
    : EQ literal
    | IN listLiteral
    ;

// -------------------------------------------------------------------------
//  WHERE / ASSUMING
// -------------------------------------------------------------------------

/** Entity/fact filter: static column conditions and/or temporal aggregations. */
whereClause
    : WHERE expr
    ;

/**
 * Counterfactual: a future-looking condition assumed true at prediction time.
 * Supports temporal aggregations and (KumoRFM) static column assumptions:
 *   ASSUMING COUNT(NOTIFICATIONS.*, 0, 7) > 2
 *   ASSUMING users.plan = 'premium'
 */
assumingClause
    : ASSUMING expr
    ;

// -------------------------------------------------------------------------
//  Expressions
// -------------------------------------------------------------------------

/**
 * Boolean expression tree. Precedence (high→low): parentheses, NOT, AND, OR —
 * encoded by alternative order (ANTLR resolves left-recursion + precedence).
 */
expr
    : LPAREN expr RPAREN                 # ParenExpr
    | NOT expr                           # NotExpr
    | expr AND expr                      # AndExpr
    | expr OR expr                       # OrExpr
    | predicate                          # PredicateExpr
    ;

/**
 * A leaf condition, or a bare value used as a (regression) target.
 *   SUM(t.c, 0, 30)                      -- bare value (regression)
 *   SUM(t.c, 0, 30) > 10                 -- comparison
 *   LAST(t.status, 0, 90) NOT LIKE '%X'  -- pattern comparison
 *   LOAN.STATUS IN ('A', 'C')            -- membership
 *   ARTICLES.DESCRIPTION IS NULL         -- null test
 */
predicate
    : valueExpr comparisonOp literal      # ComparePredicate
    | valueExpr memberOp listLiteral      # InPredicate
    | valueExpr IS NOT? NULL              # NullPredicate
    | valueExpr                           # ValuePredicate
    ;

memberOp
    : IN
    | NOT IN
    ;

/** A value: a temporal aggregation or a direct column reference. */
valueExpr
    : aggregation
    | columnRef
    ;

/**
 * Temporal aggregation over a fact column across a [start, end] window.
 * `start` is EXCLUDED, `end` is INCLUDED. Target windows are future
 * (non-negative); temporal-filter windows are past (may be negative or -INF).
 * The 4th arg is the time unit (default days).
 *   SUM(TRANSACTIONS.PRICE, 0, 30)
 *   COUNT(transactions.*, 0, 30, days)
 *   COUNT(transaction.* WHERE transaction.value > 10, -7, 0)
 *
 * The window is optional: a filtered static count omits it, e.g.
 *   COUNT(transaction.* WHERE transaction.amount > 100)   (where.md, Example 2)
 */
aggregation
    : aggFunc LPAREN aggOperand aggWindow? RPAREN
    ;

aggWindow
    : COMMA bound COMMA bound (COMMA timeUnit)?
    ;

/** The column (or `*` wildcard) being aggregated, with an optional inline filter. */
aggOperand
    : columnRef (WHERE expr)?
    ;

aggFunc
    : SUM | AVG | MIN | MAX | COUNT | COUNT_DISTINCT | LIST_DISTINCT | FIRST | LAST
    ;

/**
 * `<table>.<column>` or `<table>.*` (all rows, ignoring N/A).
 * Names may collide with keywords (a column literally named `count`,
 * `usage.count`), so both parts accept the soft-keyword set as identifiers.
 */
columnRef
    : name DOT (name | STAR)
    ;

/** An identifier, or a non-structural keyword reused as a table/column name. */
name
    : IDENTIFIER
    | softKeyword
    ;

/**
 * Keywords permitted as identifiers. Excludes the structural clause words
 * (PREDICT/FOR/WHERE/ASSUMING) and the boolean/null words (AND/OR/NOT/NULL),
 * whose reuse as bare names would make expressions genuinely ambiguous.
 */
softKeyword
    : SUM | AVG | MIN | MAX | COUNT | COUNT_DISTINCT | LIST_DISTINCT | FIRST | LAST
    | RANK | TOP | CLASSIFY | IN | IS | WITH | STARTS | ENDS | CONTAINS | LIKE
    | SECONDS | MINUTES | HOURS | DAYS | WEEKS | MONTHS
    | INF | FORECAST | TIMEFRAMES | EACH
    ;

/** A window bound: a signed integer, or (±)INF for an unbounded past/future. */
bound
    : (PLUS | MINUS)? INT
    | (PLUS | MINUS)? INF
    ;

// Time units
timeUnit
    : SECONDS | MINUTES | HOURS | DAYS | WEEKS | MONTHS
    ;

/**
 * Comparison operators. Word operators (STARTS WITH, ...) apply to text
 * columns; NOT LIKE / NOT CONTAINS are their negations.
 */
comparisonOp
    : GT | LT | EQ | EQEQ | NEQ | GE | LE
    | STARTS WITH
    | ENDS WITH
    | CONTAINS
    | NOT CONTAINS
    | LIKE
    | NOT LIKE
    ;

// -------------------------------------------------------------------------
//  Literals
// -------------------------------------------------------------------------

listLiteral
    : LPAREN literal (COMMA literal)* RPAREN
    ;

literal
    : STRING
    | DATE
    | number
    | NULL
    ;

number
    : (PLUS | MINUS)? (INT | FLOAT)
    ;

// =========================================================================
//  LEXER RULES
// =========================================================================

// ---- Keywords (case-insensitive via fragments below) --------------------
PREDICT        : P R E D I C T ;
FORECAST       : F O R E C A S T ;
TIMEFRAMES     : T I M E F R A M E S ;
FOR            : F O R ;
EACH           : E A C H ;
WHERE          : W H E R E ;
ASSUMING       : A S S U M I N G ;
CLASSIFY       : C L A S S I F Y ;
RANK           : R A N K ;
TOP            : T O P ;

// Aggregation functions
SUM            : S U M ;
AVG            : A V G ;
MIN            : M I N ;
MAX            : M A X ;
COUNT          : C O U N T ;
COUNT_DISTINCT : C O U N T '_' D I S T I N C T ;
LIST_DISTINCT  : L I S T '_' D I S T I N C T ;
FIRST          : F I R S T ;
LAST           : L A S T ;

// Boolean / comparison words
AND            : A N D ;
OR             : O R ;
NOT            : N O T ;
IN             : I N ;
IS             : I S ;
NULL           : N U L L ;
LIKE           : L I K E ;
CONTAINS       : C O N T A I N S ;
STARTS         : S T A R T S ;
ENDS           : E N D S ;
WITH           : W I T H ;

// Time units
SECONDS        : S E C O N D S ;
MINUTES        : M I N U T E S ;
HOURS          : H O U R S ;
DAYS           : D A Y S ;
WEEKS          : W E E K S ;
MONTHS         : M O N T H S ;

// Unbounded window sentinel (e.g. COUNT(t.*, -INF, 0))
INF            : I N F ;

// ---- Symbols ------------------------------------------------------------
GE     : '>=' ;
LE     : '<=' ;
NEQ    : '!=' ;
EQEQ   : '==' ;
GT     : '>' ;
LT     : '<' ;
EQ     : '=' ;
LPAREN : '(' ;
RPAREN : ')' ;
COMMA  : ',' ;
DOT    : '.' ;
STAR   : '*' ;
PLUS   : '+' ;
MINUS  : '-' ;

// ---- Literals -----------------------------------------------------------
// DATE must precede INT so 1990-01-01 is one token, not INT '-' INT '-' INT.
// Optional trailing " HH:MM:SS" (a literal space, matched explicitly).
DATE
    : DIGIT DIGIT DIGIT DIGIT '-' DIGIT DIGIT '-' DIGIT DIGIT
      ( ' ' DIGIT DIGIT ':' DIGIT DIGIT ':' DIGIT DIGIT )?
    ;

FLOAT  : DIGIT+ '.' DIGIT+ ;
INT    : DIGIT+ ;

// Single- or double-quoted string; doubled quote escapes the delimiter.
STRING
    : '\'' ( ~['\\] | '\\' . | '\'\'' )* '\''
    | '"'  ( ~["\\] | '\\' . | '""'   )* '"'
    ;

// Identifiers: table and column names (letters, digits, underscore).
IDENTIFIER
    : [A-Za-z_] [A-Za-z_0-9]*
    ;

// ---- Whitespace & comments ---------------------------------------------
WS          : [ \t\r\n]+          -> skip ;
LINE_COMMENT: '--' ~[\r\n]*       -> skip ;   // SQL-style, seen in doc examples
BLOCK_COMMENT: '/*' .*? '*/'      -> skip ;

// ---- Case-insensitive letter fragments ----------------------------------
fragment DIGIT : [0-9] ;
fragment A : [aA] ; fragment B : [bB] ; fragment C : [cC] ; fragment D : [dD] ;
fragment E : [eE] ; fragment F : [fF] ; fragment G : [gG] ; fragment H : [hH] ;
fragment I : [iI] ; fragment J : [jJ] ; fragment K : [kK] ; fragment L : [lL] ;
fragment M : [mM] ; fragment N : [nN] ; fragment O : [oO] ; fragment P : [pP] ;
fragment Q : [qQ] ; fragment R : [rR] ; fragment S : [sS] ; fragment T : [tT] ;
fragment U : [uU] ; fragment V : [vV] ; fragment W : [wW] ; fragment X : [xX] ;
fragment Y : [yY] ; fragment Z : [zZ] ;
