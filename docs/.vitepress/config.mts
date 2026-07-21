
import { defineConfig } from 'vitepress'

export default defineConfig({
  head: [
    // Tutto first-party. 'unsafe-inline' serve perche' VitePress emette
    // uno script inline per il tema e stili inline.
    [
      'meta',
      {
        'http-equiv': 'Content-Security-Policy',
        content:
          "default-src 'self'; script-src 'self' 'unsafe-inline'; " +
          "style-src 'self' 'unsafe-inline'; img-src 'self' data:; " +
          "font-src 'self'; connect-src 'self'; base-uri 'self'; form-action 'self'",
      },
    ],
  ],
    base: "/fqdn-model/",
    title: "FQDN Model",
    description: "Machine Learning FQDN Classifier",
    themeConfig: {
    footer: {
      message:
        '<a href="https://fabriziosalmi.github.io/privacy">Privacy &amp; legal</a>',
    },
        nav: [
            { text: 'Home', link: '/' },
            { text: 'API', link: '/api-reference' },
            { text: 'GitHub', link: 'https://github.com/fabriziosalmi/fqdn-model' }
        ],
        sidebar: [
            {
                text: 'Guide',
                items: [
                    { text: 'Introduction', link: '/' },
                    { text: 'Installation', link: '/installation' },
                    { text: 'Usage', link: '/usage' }
                ]
            }
        ],
        socialLinks: [
            { icon: 'github', link: 'https://github.com/fabriziosalmi/fqdn-model' }
        ]
    }
})
