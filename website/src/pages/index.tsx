import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import CodeBlock from '@theme/CodeBlock';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function Hero() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className="container">
        <Heading as="h1" className={styles.heroTitle}>
          {siteConfig.title}
        </Heading>
        <p className={styles.heroSubtitle}>
          Predictive queries over your own relational data. Declare your
          schema, wire retrievers over the storage you already have, and ask
          about the future in one line of PQL — no feature engineering, no
          training pipeline, no temporal leakage.
        </p>
        <CodeBlock language="sql" className={styles.heroCode}>
          {`PREDICT COUNT(orders.*, 0, 90, days) = 0\nFOR EACH customers.customer_id`}
        </CodeBlock>
        <p className={styles.heroCaption}>
          “For every customer, the probability they place zero orders in the
          next 90 days” — churn, as a query.
        </p>
        <div className={styles.buttons}>
          <Link className="button button--primary button--lg" to="/docs/getting-started/quickstart">
            Quickstart
          </Link>
          <Link className="button button--secondary button--lg" to="/pql/">
            Learn PQL
          </Link>
        </div>
        <div className={styles.social}>
          <iframe
            src="https://ghbtns.com/github-btn.html?user=henneberger&repo=relativedb&type=star&count=true&size=large"
            frameBorder="0"
            scrolling="0"
            width="140"
            height="30"
            title="Star relativedb on GitHub"
          />
          <iframe
            src="https://ghbtns.com/github-btn.html?user=henneberger&type=follow&count=true&size=large"
            frameBorder="0"
            scrolling="0"
            width="230"
            height="30"
            title="Follow henneberger on GitHub"
          />
        </div>
      </div>
    </header>
  );
}

function VideoStub() {
  return (
    <section className={styles.section}>
      <div className="container">
        <Heading as="h2">See it in action</Heading>
        <div className={styles.videoStub}>
          {/* TODO: replace with the real YouTube embed, e.g.
              <iframe src="https://www.youtube.com/embed/VIDEO_ID" ... /> */}
          <div className={styles.videoPlaceholder}>
            <span className={styles.videoPlayIcon}>▶</span>
            <p>Video coming soon</p>
          </div>
        </div>
      </div>
    </section>
  );
}

const PILLARS = [
  {
    title: 'A query, not a pipeline',
    body: 'PQL states the target, the population, and the time window declaratively. Change the question, change the string. Every query is validated against your schema before it runs.',
  },
  {
    title: 'Your data stays yours',
    body: 'GraphQL-style execution: all data access goes through retrievers you implement. No connectors, no credentials, no SQL generation — the same query runs on JDBC, REST, DataFrames, or a test double.',
  },
  {
    title: 'Leakage-proof by construction',
    body: 'Every retriever call carries a temporal bound, and the engine re-checks every returned row. A buggy retriever cannot leak the future into a prediction.',
  },
];

function Pillars() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className="row">
          {PILLARS.map((p) => (
            <div className="col col--4" key={p.title}>
              <Heading as="h3">{p.title}</Heading>
              <p>{p.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function RelationalTransformers() {
  return (
    <section className={`${styles.section} ${styles.sectionAlt}`}>
      <div className="container">
        <Heading as="h2">Scored by a relational transformer</Heading>
        <div className="row">
          <div className="col col--6">
            <p>
              A transformer normally attends over word tokens. A{' '}
              <strong>relational transformer</strong> attends over a small
              subgraph of your database: each token is one cell, and attention
              is masked along the structure that relates cells — same column,
              same row and its FK parents, FK children. No positional
              encodings; the schema <em>is</em> the structure.
            </p>
            <p>
              Pretrained across many schemas, it predicts{' '}
              <strong>in-context</strong>: the engine assembles the entity, its
              neighborhood, and a few labeled examples (including the entity's
              own past outcomes), and the model fills in the masked target in
              one forward pass — the relational analogue of prompting an LLM.
            </p>
          </div>
          <div className="col col--6">
            <ul>
              <li>
                <strong>vs. GBDTs on feature tables</strong> — no hand-built
                features, no per-task training, no train/serve skew.
              </li>
              <li>
                <strong>vs. graph neural networks</strong> — no per-schema,
                per-task training; one pretrained model, prompted in-context.
              </li>
              <li>
                <strong>vs. LLMs on serialized rows</strong> — typed cells,
                real keys, and real time instead of tables flattened to text.
              </li>
            </ul>
            <p>
              relativedb ships RT-J inference as a ~700-line dependency-light
              C++ engine, golden-verified against the PyTorch reference — and
              a model-free history baseline so the pipeline runs with zero
              model artifacts.{' '}
              <Link to="/docs/concepts/relational-transformers">
                Read more →
              </Link>
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

const BENCH = [
  {label: 'relativedb (CSC sampler)', seconds: 0.66, throughput: '~15,000 entities/s', pct: 1.2, highlight: true},
  {label: 'Naive per-entity pandas loop', seconds: 57.4, throughput: '~174 entities/s', pct: 100, highlight: false},
];

function Benchmark() {
  return (
    <section className={styles.section}>
      <div className="container">
        <Heading as="h2">87× faster than the naive loop</Heading>
        <p>
          Scoring 90-day churn for 10,000 customers over 200,000 orders
          (history baseline, M-series laptop). Same predictions; the CSC
          sampler indexes each table once and answers every context hop with a
          binary search.
        </p>
        <div className={styles.bench}>
          {BENCH.map((b) => (
            <div className={styles.benchRow} key={b.label}>
              <div className={styles.benchLabel}>{b.label}</div>
              <div className={styles.benchTrack}>
                <div
                  className={`${styles.benchBar} ${b.highlight ? styles.benchBarHighlight : ''}`}
                  style={{width: `${b.pct}%`}}
                />
                <span className={styles.benchValue}>
                  {b.seconds} s&nbsp;·&nbsp;{b.throughput}
                </span>
              </div>
            </div>
          ))}
        </div>
        <p className={styles.benchNote}>
          Reproduce with <code>examples/bench_naive_vs_csc.py</code>.
        </p>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout description="Predictive queries over your own relational data — PQL, retrievers, and a relational transformer.">
      <Hero />
      <main>
        <Pillars />
        <VideoStub />
        <RelationalTransformers />
        <Benchmark />
      </main>
    </Layout>
  );
}
