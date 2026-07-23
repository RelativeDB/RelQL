import type {ReactNode} from 'react';
import {useState} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import CodeBlock from '@theme/CodeBlock';
import Heading from '@theme/Heading';
import useBaseUrl from '@docusaurus/useBaseUrl';

import styles from './index.module.css';

function Hero() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className="container">
        <img
          className={styles.heroLogo}
          src={useBaseUrl('img/logo.svg')}
          alt="relativedb logo"
          width={96}
          height={96}
        />
        <Heading as="h1" className={styles.heroTitle}>
          RelativeDB
        </Heading>
        <p className={styles.heroSubtitle}>
          Predictive queries over relational data.
          Declare the shape of your relational data, wire small retrievers over
          your storage, and ask questions about the future:
        </p>
        <CodeBlock language="sql" className={styles.heroCode}>
          {`PREDICT NOT EXISTS(orders.*)\nFROM customers`}
        </CodeBlock>
        <p className={styles.heroCaption}>
          "For every customer, the probability they place zero orders"
        </p>
        <div className={styles.buttons}>
          <Link className="button button--primary button--lg" to="/docs/#quickstart">
            Quickstart
          </Link>
          <Link className="button button--secondary button--lg" to="/relql/">
            Learn RelQL
          </Link>
        </div>
        <div className={styles.social}>
          <iframe
            src="https://ghbtns.com/github-btn.html?user=RelativeDB&repo=RelQL&type=star&count=true&size=large"
            frameBorder="0"
            scrolling="0"
            width="140"
            height="30"
            title="Star relativedb on GitHub"
          />
          <iframe
            src="https://ghbtns.com/github-btn.html?user=RelativeDB&type=follow&count=true&size=large"
            frameBorder="0"
            scrolling="0"
            width="230"
            height="30"
            title="Follow RelativeDB on GitHub"
          />
        </div>
        <ClaudeDemo />

      </div>
    </header>
  );
}

function CopyCommand({lines, prompt = '$'}: {lines: string[]; prompt?: string}) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(lines.join('\n')).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <div className={styles.copyBox}>
      <code className={styles.copyText}>
        {lines.map((line) => (
          <span className={styles.copyLine} key={line}>
            <span className={styles.copyPrompt}>{prompt}</span> {line}
          </span>
        ))}
      </code>
      <button
        type="button"
        className={styles.copyButton}
        onClick={copy}
        aria-label="Copy commands to clipboard">
        {copied ? 'Copied!' : 'Copy'}
      </button>
    </div>
  );
}

function ClaudeDemo() {
  return (
    <div className={styles.claudeCard}>
      <div className={styles.claudeBar}>
        <span className={styles.claudeDots}>
          <i /> <i /> <i />
        </span>
        <span className={styles.claudeBarTitle}>Claude Code</span>
      </div>
      <div className={styles.claudeBody}>
        {/* 1. install the plugin from the marketplace */}
        <span className={styles.claudeLabel}>Install the plugin</span>
        <CopyCommand prompt=">" lines={['/plugin marketplace add RelativeDB/RelQL-cc']} />
        <CopyCommand prompt=">" lines={['/plugin install RelQL@RelQL']} />

        {/* 2. ask with the /relql command */}
        <span className={styles.claudeLabel}>Then ask</span>
        <div className={styles.claudeMsg}>
          <span className={styles.claudePrompt}>&gt;</span>
          <span>
            <span className={styles.claudeSlash}>/relql</span> which of my
            customers are about to stop ordering?
          </span>
        </div>
        <div className={styles.claudeReply}>
          <span className={styles.claudeSkillTag}>relql</span>
          <span>
            scored 12,304 customers — top risk{' '}
            <code>Jane Doe · 0.94</code>…
          </span>
        </div>
      </div>
    </div>
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

const COMPARE_COLUMNS = [
  {key: 'rt', label: 'Relational Transformer', sub: 'RelativeDB', highlight: true},
  {key: 'gbdt', label: 'GBDTs', sub: 'XGBoost & co.'},
  {key: 'gnn', label: 'Graph neural nets', sub: 'per-schema GNNs'},
  {key: 'llm', label: 'LLMs on rows', sub: 'serialized to text'},
];

type MarkState = 'yes' | 'partial' | 'no';
type Cell = [MarkState, string];

// 'yes' | 'partial' | 'no', each with a short qualifier shown under the mark.
const COMPARE_ROWS: {capability: string; cells: Record<string, Cell>}[] = [
  {
    capability: 'No hand-built features',
    cells: {
      rt: ['yes', 'raw cells'],
      gbdt: ['no', 'feature tables'],
      gnn: ['partial', 'graph wiring'],
      llm: ['partial', 'prompt design'],
    },
  },
  {
    capability: 'No per-task training',
    cells: {
      rt: ['yes', 'pretrained'],
      gbdt: ['no', 'every task'],
      gnn: ['no', 'schema + task'],
      llm: ['yes', 'pretrained'],
    },
  },
  {
    capability: 'Zero-shot on new tasks',
    cells: {
      rt: ['yes', 'in-context'],
      gbdt: ['no', ''],
      gnn: ['no', ''],
      llm: ['partial', 'if it fits text'],
    },
  },
  {
    capability: 'Schema-native structure',
    cells: {
      rt: ['yes', 'keys, rows, cols'],
      gbdt: ['no', 'flat table'],
      gnn: ['yes', 'graph edges'],
      llm: ['no', 'flattened text'],
    },
  },
  {
    capability: 'Typed cells & real time',
    cells: {
      rt: ['yes', 'native'],
      gbdt: ['partial', 'manual'],
      gnn: ['partial', 'manual'],
      llm: ['no', 'lost as text'],
    },
  },
  {
    capability: 'No train/serve skew',
    cells: {
      rt: ['yes', ''],
      gbdt: ['no', 'common'],
      gnn: ['no', 'common'],
      llm: ['yes', ''],
    },
  },
  {
    capability: 'Small & scalable',
    cells: {
      rt: ['yes', '86M params'],
      gbdt: ['yes', 'tiny'],
      gnn: ['partial', 'grows'],
      llm: ['no', 'billions'],
    },
  },
];

const MARK: Record<MarkState, {glyph: string; label: string; cls: string}> = {
  yes: {glyph: '✓', label: 'yes', cls: styles.markYes},
  partial: {glyph: '~', label: 'partial', cls: styles.markPartial},
  no: {glyph: '–', label: 'no', cls: styles.markNo},
};

function ComparisonMatrix() {
  return (
    <figure className={styles.compareFigure}>
      <div className={styles.compareScroll}>
        <table className={styles.compareTable}>

          <thead>
            <tr>
              <th scope="col" className={styles.compareCorner}>
                Capability
              </th>
              {COMPARE_COLUMNS.map((c) => (
                <th
                  scope="col"
                  key={c.key}
                  className={c.highlight ? styles.compareColHi : undefined}>
                  <span className={styles.compareColLabel}>{c.label}</span>
                  <span className={styles.compareColSub}>{c.sub}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {COMPARE_ROWS.map((row) => (
              <tr key={row.capability}>
                <th scope="row" className={styles.compareRowHead}>
                  {row.capability}
                </th>
                {COMPARE_COLUMNS.map((c) => {
                  const [state, note] = row.cells[c.key];
                  const mark = MARK[state];
                  return (
                    <td
                      key={c.key}
                      title={note || undefined}
                      className={c.highlight ? styles.compareColHi : undefined}>
                      <span className={`${styles.mark} ${mark.cls}`} aria-hidden>
                        {mark.glyph}
                      </span>
                      <span className={styles.srOnly}>
                        {mark.label}
                        {note ? ` — ${note}` : ''}.{' '}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className={styles.compareLegend}>
        <span>
          <span className={`${styles.mark} ${styles.markYes}`} aria-hidden>
            ✓
          </span>{' '}
          built in
        </span>
        <span>
          <span className={`${styles.mark} ${styles.markPartial}`} aria-hidden>
            ~
          </span>{' '}
          partial / manual
        </span>
        <span>
          <span className={`${styles.mark} ${styles.markNo}`} aria-hidden>
            –
          </span>{' '}
          not really
        </span>
      </div>
    </figure>
  );
}

function RelationalTransformers() {
  return (
    <section className={`${styles.section} ${styles.sectionAlt}`}>
      <div className="container">
        <Heading as="h2">What are Relational Transformers?</Heading>
        <p className={styles.sectionLede}>
          A transformer normally attends over word tokens. A{' '}
          <strong>relational transformer</strong> attends over a small subgraph
          of your database. The relational
          analogue of prompting an LLM, in a 86M-parameter model that scales.
        </p>
        <ComparisonMatrix />
        <p className={styles.sectionLede}>
          RelativeDB ships RT-J inference as a highly optimized,
          dependency-light C++ engine, with several quantized models for highly
          constrained environments.{' '}
          <Link to="https://arxiv.org/abs/2510.06377">Read more →</Link>
        </p>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout description="Predictive queries over relational data — RelQL, retrievers, and a relational transformer.">
      <Hero />
      <main>
        <VideoStub />
        <RelationalTransformers />
      </main>
    </Layout>
  );
}
