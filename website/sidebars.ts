import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting started',
      items: ['getting-started/installation', 'getting-started/quickstart'],
    },
    {
      type: 'category',
      label: 'Concepts',
      items: [
        'concepts/architecture',
        'concepts/relational-transformers',
        'concepts/retrievers',
        'concepts/temporal-correctness',
        'concepts/sampler-modes',
        'concepts/model-backends',
      ],
    },
    {
      type: 'category',
      label: 'How-to guides',
      items: [
        'how-to/predict-churn',
        'how-to/wire-custom-retrievers',
        'how-to/rank-recommendations',
        'how-to/forecast-demand',
        'how-to/use-native-backend',
        'how-to/choose-sampler-mode',
      ],
    },
    {
      type: 'category',
      label: 'Libraries',
      items: [
        'libraries/python',
        'libraries/java',
        'libraries/rust',
        'libraries/cpp',
      ],
    },
    {
      type: 'category',
      label: 'Contributing',
      items: ['contributing/releases'],
    },
  ],
};

export default sidebars;
