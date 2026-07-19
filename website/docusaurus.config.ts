import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'relativedb',
  tagline: 'Predictive queries over your relational data',
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
        id: 'relql',
        path: 'relql',
        routeBasePath: 'relql',
        sidebarPath: './sidebars-relql.ts',
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'relativedb',
      logo: {
        alt: 'relativedb logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/relql/',
          position: 'left',
          label: 'RelQL Language',
          activeBaseRegex: '/relql/',
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
            {label: 'Getting started', to: '/docs/#installation'},
            {label: 'RelQL language', to: '/relql/'},
            {label: 'How-to guides', to: '/docs/#predict-churn'},
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
            {label: 'Python', to: '/docs/#python-library'},
            {label: 'Java', to: '/docs/#java-library'},
            {label: 'Rust', to: '/docs/#rust-library'},
            {label: 'C++ inference engine', to: '/docs/#c-inference-engine-rtcpp'},
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
