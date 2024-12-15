import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "ArchiverateZ",
  base: "/",
  description: "A VitePress Site",
  markdown: {
    math: true
  },  
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'DSA', link: '/dsa/index'},
      { text: 'ML', link: '/aiml/ml/index'},
      { text: 'Examples', link: '/markdown-examples' }
    ],
    
    sidebar: {
      '/dsa': [
        {
          items: [
            {text: 'Getting Started', link: '/dsa/index'},
            {text: 'Exercises', link: '/dsa/exercise'},
            {text: '01-Beginning', link: '/dsa/01-beginning'},
            {text: '02-Vector', link: '/dsa/02-vector'},
            {text: '03-List', link: '/dsa/03-list'},
            {text: '04-Stack and Queue', link: '/dsa/04-stackqueue'},
            {text: '05-Sorting', link: '/dsa/05-sorting'},
            {text: '06-Tree', link: '/dsa/06-tree'},
            {text: '07-Advanced Search Tree', link: '/dsa/07-astree'},
            {text: '08-Tree Like DS', link: '/dsa/08-treelike'},
            {text: '09-Hash', link: '/dsa/09-hash'},
            {text: '10-Heap', link: '/dsa/10-heap'},
            {text: '11-String', link: '/dsa/11-string'},
            {text: '12-Graph', link: '/dsa/12-graph'},
            {text: '13-Search', link: '/dsa/13-search'},
            {text: '14-Divide and Conquer', link: '/dsa/14-dac'},
          ]
        }
      ],
      
      '/aiml/ml': [
        {
          items: [
            {text: '01-绪论', link: '/aiml/ml/index'},
            {text: '02-模型评估与选择', link: '/aiml/ml/model-evaluation-selection'},
            {text: '03-线性模型', link: '/aiml/ml/linear-model'},
            {text: '04-决策树', link: '/aiml/ml/decision-tree'},
            {text: '05-神经网络', link: '/aiml/ml/neural-network'},
            {text: '06-支持向量机', link: '/aiml/ml/svm'},
            {text: '07-贝叶斯分类器', link: '/aiml/ml/bayesian-classifier'},
            {text: '08-集成学习', link: '/aiml/ml/ensemble-learning'},
            {text: '09-聚类', link: '/aiml/ml/clustering'},
            {text: '10-降维与度量值学习', link: '/aiml/ml/dimensionality-reduction'},
            {text: '11-特征选择与稀疏学习', link: '/aiml/ml/feature-selection'},
            {text: '12-计算学习理论', link: '/aiml/ml/computational-learning-theory'},
            {text: '13-半监督学习', link: '/aiml/ml/semi-supervised-learning'},
            {text: '14-概率图模型', link: '/aiml/ml/probabilistic-graphical-model'},
            {text: '15-强化学习', link: '/aiml/ml/reinforcement-learning'},
          ]
        }
      ]
      
    },
    // sidebar: [
    //   {
    //     text: 'Examples',
    //     items: [
    //       { text: 'Markdown Examples', link: '/markdown-examples' },
    //       { text: 'Runtime API Examples', link: '/api-examples' }
    //     ]
    //   }
    // ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/acceleratez/docs' }
    ],

    search: {
      provider: "local",
      options: {
        translations: {
          button: {
            buttonText: "搜索文档",
            buttonAriaLabel: "搜索文档",
          },
          modal: {
            noResultsText: "无法找到相关结果",
            resetButtonTitle: "清除查询条件",
            footer: {
              selectText: "选择",
              navigateText: "切换",
            },
          },
        },
      },
    },
    outlineTitle: "Content",
    outline: [1,3]
  }
})
