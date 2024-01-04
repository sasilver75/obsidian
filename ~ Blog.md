
Registered **samsilver.us** using Cloudflare 

WHOIS Information
```
`Domain name: SAMSILVER.US`
`Registry Domain ID:`
`Registrar WHOIS Server: whois.cloudflare.com`
`Registrar URL: https://www.cloudflare.com/`
`Updated Date: 2024-01-04T02:24:44Z`
```

Registrant Information
```
Sam Silver
2807 Glendon Avenue
Los Angeles, CA, 90064
US
+1.2623895533
sasilver0051@gmail.com
For: Domain owner, Administrative, Technical, Billing

```

Automatic renewal: `Turned On`
This domain is set to expire on **Jan 03, 2025**
Cloudflare will automatically renew your registration 60 days before it expires.

Cost was something like `$6.50` a year.

----

Cloudflare Pages seems to be a way to host Static Sites: 
https://dash.cloudflare.com/7b7aca95efe13d9c75c25de36e7d536b/workers-and-pages/create/pages

-----
https://kinsta.com/blog/hugo-static-site/

config.toml file
- For me, I think this is the hugo.yaml file
- This is the primary configuration file, containing global settings for my site

The ==Archetypes== folder:
- Where you store content templates formatted in Markdown.
- Archetypes are especially useful if your site has multiple content formats.
- With HugoArchtypes you can create a template for each content type on your site, allowing you to pre-populate generated Markdown files with all the necessary configuration settings.
- These may seem complex and unnecessary at first, but can end up saving a lot of time in the long run.

The ==Content== folder:
- The content folder is where the actual post content goes.
- Hugo supports both Markdown and HTML formats, with Markdown being the more popular option.
- Hugo treats each top-level directory in the content folder as a content section -- content sections in Hugo are similar to "custom post types" in Wordpress; For example, if you site has posts, pages, and podcasts, the content folder would have posts, pages, and podcast directories where content files for these different post types would live.

The ==Layouts== folder:
- The layouts folder contains HTML files that define the STRUCTURE of your site.
- In some cases, you might see a Hugo site without a layouts folder, because it doesn't have to be in the projects root directory and can instead reside within a theme folder instead.
- Similar to Wordpress themes that use PHP for templating, Hugo templates consist of base HTML with additional dynamic templating powered by Golang's built-in `html/template` and `text/template` libraries. The various HTML template files required for generating your site's HTML markup are in the layouts folder.

The ==Themes== folder
- For sites that prefer a more self-contained way of storing template files and assets, Hugo supports a themes folder.
- Hugo themes are similar to Wordpress themes in that they're stored in a themes directory and contain all of the necessary templates for a theme to function 
- some Hugo users prefer keeping theme-related files in the project's root directory, but storing the files within the themes folder allows for easier management and sharing.

The ==Data== folder
- This is where you can store supplemental data (JSON, YAML, TOML format) that is used to generate site pages.
- Data files are beneficial for larger data sets that might be cumbersome to store directly in a content or template file.
- For example, if you wanted to create a list of USD inflation rates from 1960 to 2020, it would take around 80 lines to represent the data (one line for each year)... Instead of putting this data directly in the content or template file, you can create it in the dat folder and populate it with the necessary information.

The ==Static== folder
- Hugo Static folder is where you store static assets that don't require any additional processing.
- This is typically where users store Images, fonts, DNS verification files, and more.
- When a Hugo site is generated and saved to a folder for easy deployment, all files in the static folder are copied as-is.
- (If you're wondering why we didn't mention JS or CSS files, it's because they're often dynamically processed via pipelines during site development. In Hugo, JS and CSS files are commonly stored in the `theme` folder because they require additional processing.)

How to add a theme to a Hugo site
- Installing a premade theme is a great way to get started!
- Navigate ot the project's theme folder:
```
cd <hugo-project-directory>/themes
git clone https://github.com/spf13/hyde.git

...Then, in your config.toml (or similar config file)...
theme = "hyde"
```

How to previe a Hugo site:
- Hugo ships with an integrated webserver for development purposes
- To start Hugou's webserver, run this in the root of your project:

```
hugo server -D
```
Huge them builds your site pages and makes them available at http://localhost:1313/

By default, Hugo's local development server will watch for changes and rebuild the site automatically -- since Hugo's build speed is so fast, updates to your site can be seen in near-real-time -- something that's rare to see in the SSG world!


Adding content to a Hugo site:
- Adding content to a Hugo site is different frmo full-fledged CMSs like WordPress or Ghost -- with Hugo, htere's no built-in CMS layer to manage your content; you're expected to manage and organize things as you see fit!
	- So there's no explicitly "correct" way to do content management in Hugo.

### Content Sections in Hugo
- In Hugo, the first content organization tool at your disposal is the ==content section== -- this is similar to a post type in Wordpress. Not only can you use it as a content filter, but you can use it as an identifier when creating custom themes.

For example, fi you have a **blog** content section folder, you can use it to store all your blog posts and render a specific page template that only applies to blog posts.

### How to add Posts in Hugo
- Let's create a content section for blog posts and add a few pieces of content. Create a new folder named **posts**  in our project's content folder!
- Let's create another organization layer inside the posts folder by adding a 2024 folder...
















