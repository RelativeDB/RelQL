import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  pqlSidebar: [
    'index',
    'tutorial',
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/query-structure',
        'reference/aggregations',
        'reference/conditions',
        'reference/task-types',
      ],
    },
    'cookbook',
  ],
};

export default sidebars;
