import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// The engine guide is a single document; its headings drive the on-page table
// of contents, so the sidebar has nothing to nest.
const sidebars: SidebarsConfig = {
  docsSidebar: ['intro'],
};

export default sidebars;
