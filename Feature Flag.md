A runtime switch that turns behavior on or off without deploying new code.

Teams use feature flags to roll out features gradually, test variants, disable risky changes quickly, or expose functionality only to certain users, accounts, regions, or environments.

```
if (flags.newCheckoutExperience) {
	showNewCheckout();
} else {
	showOldcheckout();
}
```
The main risk is ==flag debt==; old flags can pile up and make the code harder to understand if they are not removed after the rollout!


Oftentimes implemented as a small decision layer around code paths:
```
if (isEnabled("new_checkout")) {
    return renderNewCheckout();
}

return renderOldCheckout();
```

# Feature Flag Types
- Boolean flags: on/off
- Percentage rollouts: enable for 1$, 10$, 50%, etc.
- Targeting rules: Enable for specific users, teams, plans, regions, or environments
- Variants flags: Return "control", "variant_a", "variant_b", for A/B tests
- Server-side flags: Safer for backend logic, permissions, billing, data changes
- Client-side flags: Useful for UI changes, but shouldn't protect sensitive behavior by themselves, since they're client-side, which is somewhat uncontrolled.

# Implementations
The `isEnabled()` check can be backed by different systems:

##### 1) Static Config
- Flags live in environment variables, config files, or build-time settings.
```
NEW_CHECKOUT_ENABLED=true
```
- Good for simple ops toggles, but changes usually require a redeploy or start.

##### 2) Database-backed Flags (You manage)
- Flags are stored in a database or admin-managed config table. It might look something like:
```
flag_name: new_checkout
enabled: true
rollout_percentage: 25
allowed_users: [...]
```
- The app reads this config and decides whether the feature is active.

##### 3) Remote Flag Service (You buy a managed service)
- Teams often use tools like [[LaunchDarkly]], [[Statsig]], Unleash, Split, or a homegrown service. The app uses an SDK:
```js
const enabled = flags.enabled("new_checkout", {userId, accountId, plan, region})
```
- This allows for targeting, gradual rollout, experiments, audit logs, and fast rollback.

