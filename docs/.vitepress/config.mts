import { defineConfig } from 'vitepress'

export default defineConfig({
    title: "FQDN Model",
    description: "Machine Learning FQDN Classifier",
    themeConfig: {
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
