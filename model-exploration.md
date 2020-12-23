# Model Exploration

## Distance measure

In these approaches **distance measures** are used in some way to get the match.

### Unsupervised: Clusters

#### Clusters of Openings

| Cluster | Openings |
| :--- | ---: |
| 1 | 1, 2, **3**, 4, 10 |
| 2 | 7, 9, **12**, 8 |
| 3 | 5, **11**, 6 |

> The **bolded openings** are the representatives of each cluster

To create the clusters there has to be a **transformer to points** for `Openings`, so we can represent them as points.

The clusters need to be updated once some `Opening` is changed, deleted or created.

When a `User` is created/edited in order to have a match, this is what happens:

1. The `User` is measured against the representative of each cluster.

   > A **measure of distance** is needed between an `Opening` \(point\) and a `User`

2. The step above yields the distance from the `User` to each of the Clusters. For the Cluster\(s\) with the least \(1/2/X?\) distance\(s\):
3. For each `Opening` of the Cluster, calculate the distance between it and the `User`.
4. The closest match\(es\) is\(are\) returned.

A threshold for step 2 could be used.

An example of this:

`User` a = ... Some user that is really talented in Machine Learning

For each cluster, we get a distance \(or %?\):

| Cluster | Distance |
| :--- | :--- |
| 1 | 5 |
| 2 | 2 |
| 3 | 1 |

It's clear that the `Openings` of Cluster 3 would interest `a` more.

So, for each `Opening` in Cluster 3 we calculated another distance \(or %?\):

| Cluster | Distance |
| :--- | :--- |
| 5 | 1.2 |
| 11 | 1 \(same as before\) |
| 6 | 0.9 |

Now we can return all 3 openings \(6, 11 and 5\) or all that are above some threshold.

### Simple

A simple approach \(to implement\) would be to calculate the distance between each `Opening` and each `User`.

This could be slightly optimized if it's observed that lots of "different" `Users` have the exact same qualities, so instead of doing the calculation for each `User` we could use representative `Users`. The same for `Openings`.

### Example

![Model Interaction Example](.gitbook/assets/group-79.png)

## Supervised approach

Here we would need to fabricate data.

What would be the features and the output? Classification?

