
import { defineConfig } from 'vitepress'

export default defineConfig({
  head: [
    // Everything this site loads is first-party. 'unsafe-inline' is required
    // because VitePress emits an inline appearance script and inline styles.
    // Applied to the built site only: `vitepress dev` serves HMR over a
    // websocket, which a strict connect-src would block as soon as the dev
    // server is not same-origin (--host, or a custom server.hmr.port).
    ...(process.env.NODE_ENV === 'production'
      ? [
          [
            'meta',
            {
              'http-equiv': 'Content-Security-Policy',
              content:
                "default-src 'self'; script-src 'self' 'unsafe-inline'; " +
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; " +
                "font-src 'self'; connect-src 'self'; base-uri 'self'; form-action 'self'",
            },
          ] as [string, Record<string, string>],
        ]
      : []),
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
