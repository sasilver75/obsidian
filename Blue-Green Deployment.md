
A release strategy where you run ==two production-like environments==:
- ==Blue==: The current live version serving users
- ==Green==: The new version that you deploy and test, while blue keeps serving traffic.

==Once Green is ready, you switch traffic from Blue to Green==, usually through a [[Load Balancing|Load Balancer]], [[Routing|Router]], [[Domain Name Service|DNS]], or platform routing control. If something breaks, you can quickly roll traffic back to Blue.
- +: Fast rollback, low downtime, safer releases
- -: You need duplicate infrastructure, good routing/health-check controls

Example flow:
1. Users are on Blue
2. Deploy the new app version to Green
3. Run [[Smoke Test]]s against Green
4. Switch traffic to Green
5. Keep Blue around temporarily as rollback


# How do we accomplish it?
- AWS: [[[Amazon CodeDeploy|AWS CodeDeploy]] for [[Amazon EC2|EC2]]/[[Amazon Elastic Container Service|ECS]]
- [[Kubernetes]]: Either "Argo Rollouts" or "Flagger" can automate blue/green and [[Canary Release]]s instead of making you manually switch Services/Ingress.
- Google Cloud: [[Google Cloud Run]] revisions with traffic splitting, or [[Google Cloud Build]]/[[Google Cloud Deploy]] for more structured pipelines.
- Vercel: Rolling Releases can route a configured percentage of users to a new deployment before full promotion.

Otherwise you have to do it yourself with NGINX/HAProxy/Envoy/Terraform/etc.