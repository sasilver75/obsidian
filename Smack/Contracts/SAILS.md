
SAILS is an acronym rolled by John Falcone while he was still working at DIU before we hired him to work at Smack.
- Also called "THE NIWC PAC Thing"




CJ Scrapper, Aug 13 All Hands
> - 72 hour OpEval sometime in September

^ I believe this is also called Milestone 3.


Re: Database thing

The NC3 folks @ NIWC (Nuclear command control etc) are using Neo4j, but likely a highly pinned older version... and the idea is that maybe we don't use any of that, so can I get on AuraDB, make a new image at that pinned version, and dumb the DB into it, and test our software against it.
Obviously any comms with the government 

If that CRADO pops off and we need to do graph resolvers...


___________

# Call on August 17 with the OPE Engineers


We're trying to get into IL6, yes.
In IL6, OSA is going to remain functional for at least several months, through the end of March 2027.
The unclassed side is actively being taken down though... if you're trying to do something before September 15th, you can point to the ____ server...
We're working with the Orion team to set up an SRM instance.
...You can get started with the one in OSA Tools, push the scan results, and do the mitigations.... and at some future point before September 15, we'll be... putting out an announcement as to when the new 

OSA: An environment that provided a lot of developer tooling; a Git repo, bitbucket, confluence, along with static/dynamic scanning systems, the SRM for reporting, and those types of tools.
Orion: Was another environment that had some of that stuff. Were using Gitlab instead of... the Atlassian set of tools that OSA was using, so navy decided to consolidate to Orion... under the belief that we had duplicative functionality. but of course when you have two different systems that appear to be duplicative, you often have gaps; SRM was one of those things.

Q: So are we still emailing a list of containers to Nate and he's running the scans, or can we submit them on our own?
A: In OSA, you'd have put the thing into artifactory, and then... might have had JFrog XRay which would automatically scan once you upload. But you can also...in a tektonic pipeline, kick off an XRay scan, which is really just looking through the index of that artifact and doing a database search against its vulnerability database and doing a match. It's a little different from a source code scan, where it evaluates source code... it's a lookup based on your software components and their version numbers. So if you're using lib-hello v5, and there's a vulnerability there, it will come up.

All of it is required... it depends on whether your containers are vendor-proprietary... you can get a waiver on the source ocde scans for those, but hte other ones will still apply, a vulnerability scan against the container to check for vulnerable dependencies. Will still need to do secrets detection scan as well. When you upload into artifactory, it generates an S-bomb (SBOM, Software Bill of Materials?) ... and the last one is doing an OWASP Zap scan against your exposed web interfaces. Those all get ingested back into SRM.

Q: Yeah, after submittin to artifactory, they get scanned, then posted to SRM for comments, yeah.
A: So yeah, you're uploading your own containers, right? Not building them in OSA or Orion?
Q: Yeah, prebuilt
A: Yeah, Orion also has artifactory. OSA has OpenShift Tekton, Orion uses Gitlab as their primary devsecops pipeline. They have both Gitlab community and Gitlab ultimate. From a scanning standpoint, there are a couple options. You can use secrets detection in Gitlab, or... can use something like __hog to do.

Can run your TrippleHogg against it yourself in your own local environment
Q:  Yeah, we're using Trippy right now...
A: Yeah, some type of secrets detection, vulnerability scan, etc... you should be good.

Q: Yeah, we get a different set of results back.

Q: Accessing the SRM in OSA tools.... if we can't get up into OPE until we pass gate 6, is there like a holding area? Or I don't know what the right word would be... where we can't access that SRM and make those notes?
A: You don't have to be in OPE Dev to start working on SRM. Once you have OSA accounts and have a project setup, you can go ahead and start importing your containers into artifactory, uploading help charts into artifactory, etc. As far as project onboarding... they will create a project for you in SRM including any child projects you might need based on your project... You can look at that at any point, once you've run your Tekton pipeline in OSA and pushed to OSM. It's supposed to be an iterative process...

Q: we've already migrated Confluence and Jira... unfortunately you need accounts on both environments... You'll need 


What is COSMOS?


If you wanted to do the Neptune thing...

Security Groups
VPC Peering or AWS Transit Gateway
- VPC peering: A way of connecting different VPCs together; the old-school way of doing it, but it's mesh... you have to do it for each VPC you want to connect, making a spiderweb that can be quite a mess.
- Transit Gateways are more of a hub-and-spoke capability; very cool, but there's a weakness of them too: There's not an easy fine-grained way to filter traffic like you could on a firewall/router (e.g. ACLs or firewall rules saying X subnet can only talk to Y subnet on Z ports...)... transit gateway can't do that.
- In COSMOS, until recently... they had it wide-open until recently, but n now locked down... and they tell  people to use VPC peering if you have a product-to-product requirement, whereas north-south is still handled by the transit gateway. For IL6, we only do Transmit Gateway... but now I'm starting to second-guess what we're doing in IL6. So there might be some small deltas like that that might be confusing, and those deltas do exist, but at a high level... the basic ideas should translate from OP Dev to IL6.
If AWS Roles get in involved, we have to do the AWS cross-account role thing.



> So it sounds like the somewhat onerous process for using Neptune would be different for both the development (OP Dev) and the IL6 environments... but also that "it should translate pretty well."


https://www.cosmos.navy.mil/
Is where you go to create identities.
app.cosmos.navy.mil
Is where you go to actually access Cosmos.

Between these is how you would create another product and get access into it.


Talking to Vince: 
1. So it sounds like we can't CURRENTLY upload our stuff to Artifactory and have it be automatically scanned... but we can soon, maybe?
2. 

> A few transcription corrections: “S-bomb” is **SBOM**, “OAuth Zap” is **OWASP ZAP**, “Tektonic” is **Tekton**, “Trippy” is **Trivy**, and the secrets tool was likely **TruffleHog**.