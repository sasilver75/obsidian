



Why does Frontend have 70,000 lines of code?
- JSX is verbose
- Frontend repo holds all the UI code while other services are more focused services.

Microservices: Our services are more like small-to-medium services. We split them when they get too large....We're intentional in there being single entrypoint into the cluster.

We aren't afraid of having a single service do stuff that's adjacent to core work; we aren't super aggressive about each service having a single API endpoint or anything like that.


Single ingress point.... constrains some of our ability to work with third party tooling. It would be great if .. we had any of the creepy end user telemetry setups, for instance... those things are very great from a UI design perspective, etc.... we're never going to be able to spy on our users like we would be if we were trying to be Google or whatever it might be. 

UI is served as a SPA (compiled JS that then communicates via HTTPS and Web Sockets) to the clster ingress controller, which hits an NGINX server, which does routing to different backend servers depending on what's going on. 

There's three main things to call out.... The NGINX server serves the JS Bundles that are used to bootstrap  everything else
If you're just trying to load the UI, the ngingx responds with some JS, CSS, images, and then everything executes on the client.
When we get into the data flow, most of that goes through GraphQL... and so the web application in the browser submits GraphQL requests through Nginx to Masetro, our main API gateway....
The big other use case is for signin, which goesons the /auth path, which gets directed to Spicy, our authn server. IT's important that the authn server is split from the others servers for.... technical cybersecurity reasons.


3/4 clusters
Dev: dev.smackdev.com
Staging: staging.smackdev.com
Demo: demo.smackdev.com: Our commercial demo instance
Gov: demo.smackgov.com: 

Dev/Staging/Demo all live in our AWS commercial account (for right now)
When we were setting things up by default, AWS commercial instances are easy to plug into , and we kinda started there. We also had an AWS GovCloud account which is an offering from AWS approved for government/military workloads with provisional authority up to IL5... we probably want to to move dev/staging up to IL5.

All four of these are Kubernetes services we run in EKS... a managed Kubernetes system that AWS gives.
WE don't really use much of any of the fancy AWS features... that's an intentional decision by us... we want to be as environment-agnostic as possible, which is why we built the bullet on containerizing things early and building on K8s so that things are portable.
So if we have to deploy to an Azure or an OpenShift environment for [[Defense Innovation Unit|DIU]]. We use EKS because... it's easier than manually spinning up a K8s service ourselves, but we don't use any of the fancy EKS-specific features.
- We don't do any autoscaling right now. We never will need to, probably.


Q: Why do we have these four different clusters, what do they do?
A: Demo and Gov are basically our pseudo-production environments. These are the ones that all of the BD and growth guys use to demo our product to customers. They're constantly running demos; internally, we need to make sure that they're available 24/7, since Ray will be flying back from Hawaii, getting drinks at an airport bar, and run into an Admiral.
- To facilitate that, we pin all of the services versions that are runnign in Demo and Go to specific version numbers of the different services... so when you merge code to main in any of our repos, it automatically build s a new version of that service as a container, and push to our container registry.... that container will have its own version identifier associated with it. For the dev cluster.... every time you do that, it actually redeploys the dev environment to pull in the latest version of the containers for ALL the services (in dev; which exists to be a development aid).
- Dev database gets wiped (or redeployed?) too.... exposes some issues where when you were working on something locally, you have ...
	- We do some convenience feeding from a set script.
- Because the dev cluster reboots whenever anyone does anything, it's inconvenient for doing elaborate testing flows
	- We thus created the ==staging== environment, which has the behavior as demo and gov with the pinned versions, etc.
		- None of the BD/growth guys use the staging environment, so we can break it, redeploy it at will ,etc. It's sort of middle ground between dev

Demo vs Gov: Demo is in commercial AWS, Gov is in Govcloud, but there are interests in 


https://cyberintelsystems.com/classification-codes/
Classification codes

Sometimes people will use these colos as stand-ins for impact level
- Unclassified: ~IL4
- CUI is kind of like protected, the purple banner.... that's IL5...
- Secret... IL6
- Top Secret... Orange or  gold.x

Data about enemy capabilities is generally secret
Data about our capabilities is generally top secret



Predictive ISR: Teh JFN contract
COA Genreation: Also sometimes referred to as P2C, Plan to CONP, built for the MCWL contrct, we got a letter of successon that 
SAILS is for DIU
Typohoon is the product UI we built for the 82nd ABN we built of the guys in Jordan a few months ago
Athena is for KG ... access
Agent Workbench is SAGE, which exists to stitch together different LLM agentes
CLEAR is for scenario scripting that is an input into our ... RL training efforts, an internal-only product.
User Admin is user administration, where you can set up usersa nd permission roles and assign them things.



![[Pasted image 20260813153414.png]]
We have the ability to create users either HERE or using external identity providers, which would usually be  

Spicy is the abstraction layer that captures all this complexity; this is ...



JFN/JADO ISR:
- They wanted us to integrate with a weaponeering system they had that did that piece, and they wanted us to just handle the ISR component.
	- A mixture of drones (MQ-4C, RQ-4B, Satellites, etc.)
	- Hawkeye 360: ELINT
	- Worldview: EO/IR
	- RadarSat: SAR