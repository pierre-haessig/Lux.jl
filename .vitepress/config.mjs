import { defineConfig } from "vitepress";
import { generateSidebar } from "vitepress-sidebar";
import mathjax3 from "markdown-it-mathjax3";
import markdownitsup from "markdown-it-sup";
import markdownitabbr from "markdown-it-abbr";
import markdownittasklists from "markdown-it-task-lists";
import { readdirSync } from "fs";

const customElements = ["mjx-container"];

const sideBarGeneratorList = [];

const VITEPRESS_DOC_VERSIONS = readdirSync(".", { withFileTypes: true })
  .filter((x) => x.isDirectory())
  .map((x) => x.name)
  .filter((x) => x.startsWith("v"))
  .filter((x) => !["v0.5.0", "v0.5.1", "v0.4.58", "v0.3.0"].includes(x));
VITEPRESS_DOC_VERSIONS.push("dev");

for (let i = 0; i < VITEPRESS_DOC_VERSIONS.length; i++) {
  var version = VITEPRESS_DOC_VERSIONS[i];
  sideBarGeneratorList.push({
    documentRootPath: "/",
    scanStartPath: "/" + version + "/tutorials/",
    useTitleFromFileHeading: true,
    capitalizeFirst: true,
    rootGroupText: "Tutorials",
    manualSortFileNameByPriority: ["beginner", "intermediate", "advanced"],
    resolvePath: "/" + version + "/tutorials/",
    includeFolderIndexFile: true,
    rootGroupLink: "/" + version + "/tutorials/",
  });
  sideBarGeneratorList.push({
    documentRootPath: "/",
    scanStartPath: "/" + version + "/introduction/",
    useTitleFromFileHeading: true,
    rootGroupText: "Getting Started",
    resolvePath: "/" + version + "/introduction/",
    manualSortFileNameByPriority: [
      "installation.md",
      "quickstart.md",
      "overview.md",
      "resources.md",
      "citation.md",
    ],
    rootGroupLink: "/" + version + "/introduction/",
    includeFolderIndexFile: true,
  });
  sideBarGeneratorList.push({
    documentRootPath: "/",
    scanStartPath: "/" + version + "/manual/",
    rootGroupText: "Manual",
    useTitleFromFileHeading: true,
    resolvePath: "/" + version + "/manual/",
    includeFolderIndexFile: true,
    manualSortFileNameByPriority: ["interface.md"],
  });
  sideBarGeneratorList.push({
    documentRootPath: "/",
    scanStartPath: "/" + version + "/api/",
    rootGroupText: "API Reference",
    useTitleFromFileHeading: true,
    resolvePath: "/" + version + "/api/",
    includeFolderIndexFile: true,
    rootGroupLink: "/" + version + "/api/",
    manualSortFileNameByPriority: [
      "Lux",
      "Building Blocks",
      "Domain Specific Modeling",
      "Accelerator Support",
      "Testing Functionality",
      "layers.md",
      "utilities.md",
      "flux_to_lux.md",
      "contrib.md",
    ],
    hyphenToSpace: true,
    underscoreToSpace: true,
  });
}

const sideBar = generateSidebar(sideBarGeneratorList);

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "LuxDL Docs",
  description: "Elegant Deep Learning in Julia",

  lastUpdated: true,
  cleanUrls: true,

  ignoreDeadLinks: true,

  themeConfig: {
    externalLinkIcon: true,

    logo: {
      light: "/lux-logo.svg",
      dark: "/lux-logo-dark.svg",
    },

    nav: [
      { text: "Home", link: "/" },
      { text: "Getting Started", link: "/dev/introduction/" },
      { text: "Ecosystem", link: "/dev/ecosystem" },
      { text: "Tutorials", link: "/dev/tutorials/" },
      { text: "Manual", link: "/dev/manual/interface" },
      {
        text: "API",
        items: [
          { text: "Latest", link: "/dev/api/" },
          { text: "Stable", link: "/stable/api/" },
          { text: "v0.5", link: "/v0.5/api/" },
        ],
      },
    ],

    sidebar: sideBar,

    socialLinks: [
      { icon: "github", link: "https://github.com/LuxDL/" },
      { icon: "twitter", link: "https://twitter.com/avikpal1410" },
    ],

    search: {
      provider: "local",
    },

    footer: {
      message:
        "Released under the MIT License. Powered by the <a href='https://www.julialang.org'>Julia Programming Language</a>.",
      copyright: "Copyright © 2022-Present Avik Pal",
    },
  },

  head: [
    [
      "script",
      { async: "", src: "https://www.googletagmanager.com/gtag/js?id=G-Q8GYTEVTZ2" },
    ],
    [
      "script",
      {},
      `window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-Q8GYTEVTZ2');`,
    ],
  ],

  markdown: {
    theme: {
      light: "github-light",
      dark: "github-dark",
    },

    config: (md) => {
      // TODO: https://github.com/luckrya/markdown-it-link-to-card
      md.use(markdownitsup);
      md.use(markdownitabbr);
      md.use(markdownittasklists);
      md.use(mathjax3);
    },
  },

  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => customElements.includes(tag),
      },
    },
  },
});
