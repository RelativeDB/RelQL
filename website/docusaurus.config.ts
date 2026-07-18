import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'relativedb',
  tagline: 'Predictive queries over your own relational data',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://relql.com',
  baseUrl: '/',

  organizationName: 'RelativeDB',
  projectName: 'RelQL',

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'pql',
        path: 'pql',
        routeBasePath: 'pql',
        sidebarPath: './sidebars-pql.ts',
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'relativedb',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/pql/',
          position: 'left',
          label: 'RelQL Language',
          activeBaseRegex: '/pql/',
        },
        {
          href: 'https://github.com/RelativeDB/RelQL',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {label: 'Getting started', to: '/docs/getting-started/installation'},
            {label: 'RelQL language', to: '/pql/'},
            {label: 'How-to guides', to: '/docs/how-to/predict-churn'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub', href: 'https://github.com/RelativeDB/RelQL'},
            {label: 'Issues', href: 'https://github.com/RelativeDB/RelQL/issues'},
            {label: 'Discussions', href: 'https://github.com/RelativeDB/RelQL/discussions'},
          ],
        },
        {
          title: 'Libraries',
          items: [
            {label: 'Python', to: '/docs/libraries/python'},
            {label: 'Java', to: '/docs/libraries/java'},
            {label: 'Rust', to: '/docs/libraries/rust'},
            {label: 'C++ inference engine', to: '/docs/libraries/cpp'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} relativedb. Apache-2.0.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['java', 'rust', 'sql', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
