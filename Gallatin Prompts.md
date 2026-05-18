```
Let's pause on the questions for a moment, I want to talk you through some of the views that I see on the actual Gallatin product, Navigator, which will inform how we build our product. Not every feature from the Gallatin product needs to be in ours, however. This will be a discussion.

[Image #1] This is an image from the demo that showcases a high-level overview of the 3rd Infantry Bridge Combat Team (BCT).

You can see that it has a location of "51P UQ 0232 7336". This is a reference using the MGRS (Military Grid Reference System) that resolves to somewhere near Calapan City, Phillippines.

You can see that it has 3671 personnel associated with it.

You can see that there's an "Offense" tag associated with the BCT.

You can see that this view has an "Overview", "Orders", and "Config" tabs. The overview tab is currently selected.

It seems that there are four subsections to the Overview tab: LOGSYNC Matrix, Movements, Inventory, and Reporting Units.

Currently, it seems that LOGSYNC Matrix and Movements sections are collapsed. It seems that the Inventory and Reporting Units sections are expanded.

Under inventory, we see different inventory items ("Bulk Water", "MRE", "Turbine Fuel", "[C995] M136")... Each of these items has a series of boxes next to it that describe the inventory state of that item... The first box says OH (assumedly "On Hand"), then the following boxes 24, 48, 72, and 96 (assumedly hours in the future). Each box is colored according to the "BRAG" color coding commonly used in military logistics (Black, Red, Amber, Green, in ascending amount of supplies. These colors are typically used in Logistic Status (LOGSTAT) reports to give commanders snapshots of what units need immediate resupply, and which are self sufficient. Green means fully mission-capable or stocked (e.g. 85-100% of required supplies), Amber means reduced capability (e.g. 70% to 84%), Red means combat ineffective or heavily restricted (e.g. 50% to 69%), and Black means mission critical or out of stock, and at grave operational risk (<50%, e.g.).

Under Reporting Units, we have items like "2-27 IN", "3-4 CAV" (meaning 3rd Squadron, 4th Cavalry Regiment). One of these items seems to have a "ME" tag next to it, perhaps indicating the the current identity interacting with the application is attached to 3-4 CAV. Each group in this situation has little boxes that are labeled I, II, V, and PAX. I assume that I/III/V are "Classes of Supply," (e.g. Class I is Subsistence, meaning food/water/rations), whereas "PAX" might mean "Passengers/Personnel." Again, they use the same GARB system.

See on the map that we have a variety of entities pictured scattered around the map, each with pins that assumedly symbolize their role... 
You can see: 
- "HHC, 3IBCT": Headquarters and Headquarters Company, 3rd Infantry Brigade Combat Team
- "3IBCT 25ID": 3rd Infantry Brigade Combat Team, 25th Infantry Division
- "2-27 IN": 2nd Batallion, 27th Infantry Regiment
- "2-35 IN": 2nd Battalion, 35th Infantry Regiment
- "3-4 CAV": 3rd Squadron, 4th Calvary Regiment...also has a "ME" tag on the pin
- "3-7 FA": 3rd Batalion, 7th Field Artillery Regiment
- "65 BEB": 65th Brigade Engineer Batallion
  
I feel that even with this simple view, a lot has been communicated about the application. 

Notes from the video demonstration:
- When the demonstrator clicks on the "3IBCT 25ID" pin, the map zooms in to focus on that pin.
- The left side is referred to as the Logistics Common Operating Picture.
- The color coding is "classic" in defense logistics, notes that you can set thresholds for what the colors mean. 
- "You're generally trying to optimize for the unit having Days of Suppleis for roughly three days; that's kind of the rhythym about how supplies are planned, and you want to make sure that with current supplies, they can sustain themselves for three days."
```

Image 1 ![[Pasted image 20260517154305.png]]




________

```
While in the view of the 3rd Infantry Brigade Combat Team, upon clicking on one of the Reporting Units names in the overview tab, such as the "3-4 Cav" button, 

we see [IMAGE 1]
The left information interface changes to focus on 3rd Squadrom, 4th Cavalry Regiment, the map quickly zooms out, pans, and zooms in to the "3-4 CAV" pin.
The left information interface shows:
- The unit title, "3rd Squadron, 4th Cavalry Regiment"
- The location of the unit, 51P UQ 1308 4023
- The headcount of the unit , 440
- Tags: "Offense", "Main Effort"

There's an "Overview" and "Config" tab. The Overview tab is selected.

Under this selected Overview tab, there are "Movements", "Inventory", and "LOGSTATs" sections. Movements and LOGSTATs are collapsed. The LOGSTATs item shows a "24h" bubble next to it, though the meaning of this is unclear. Inventory is expanded, and shows:
- "MRE"
- "Bulk Water"
- "Turbine Fuel, ..." (assumedly title is cutoff)
- "[AB57] CTG, ..."
- "[C995] M136 ..."
- "[C141] FGM-1..." 
  
For each, it shows the OH, 24, 48, 72, 96 "blocks", each color-coded using the GARB/BRAG colors. 
```

Image 1
![[Pasted image 20260517154610.png]]


___________


```
[IMAGE 1]
It seems that by clicking on the "inventory" area of the left pane (not on any specific line item within it, just anywhere in it), that we open up this expanded inventory information pane. 

The expanded inventory information pane says "Inventory"
and has three tabs by which you can view information, it seems:
- Class
- Status
- Unit
  
Class is currently selected. 
There are an "Edit" and "Add Items" button in the top-right of the expanded inventory imnformation pane.

Below the header sction, see that this shows, for each class (each having their own unique icon, next to the class), the inventory items, and their status. 

The columns for each Class of Supply table are: 
- (unlabeled): Shows Stars that are either filled in or not. Not clear if this is some sort of derived data that means "important," or if this is some sort of toggleable "favorite"-like star icon for the current operator to toggle/on off.
- Resource: The resource in question, e.g. "Bottled Water", "MRE", "Hydraulic Fluid MIL-PRF-83282", "[AB43] CTG., .300 WIN MAG"
- U/I (assumedly meaning Unit of Issue, which specifies the exact grouping or packaging in which each product is ordered): Some of these values are "Gallon", "Round", while others are "Case [24]", where I suspect that Case [24] means that there are 24 items to a case; This [24] would then be the "Quantity per Unit Pack". 
- On Hand: The current, on-hand inventory in the unit, in terms of the unit of issue. "2,015", "12,092", "25". The box is colored according to BRAG.
- +24h: The inventory that the unit is expected to have in 24h, assumedly colored using BRAG as "On hand" is (may be a different color, etc).
- +48h: Same logic as "+24h"
- +72h: Same logic as "+24h"
- +96h: Same logic as "+24h"
  
The dialogue from the demo:
"We're seeing here an overview of that unit's inventory on hand, what they expect to have in 24h, 48, and so on. These are not actual values, these are model predictions from Navigator, and can obviously be overwritten by a huamn at any point in time, the human is in control."
```

Image 1
![[Pasted image 20260517171023.png]]


_____________

```

"Let's say we just heard over the radio that their MRE got totally wiped out (he edits the On Hand MRE number to 0, from 333)... We save this, and then we see... that we're regenerating courses of actions. This is usually done over hours by our team, and we already see that Course of Action is updated. This is usually done over the course of hours by a whole team, and we're already seeing that courses of action are updated. We obviously need to supply them..."

[IMAGE 1]
So see that pressing the edit button makes the tables sort of grayscale, and then the on-hand column is darker, indicating that that is the sole editable column. So logistics officers can manually change the current on-hand supply of a given item in the inventory of a unit, but they can't change the 24 or 48 or 72 hour amounts of supply. You can also see that there is an X that appears to the right in its own column at the right side of the table, which allows for you to remove an inventory item. 

The operator updates the "On Hand" quantity of the MRE inventory item to 0, and pressses the Save button.


[IMAGE 2]
[IMAGE 3]
You can see from these sequential images that saving of the updated on-hand inventory causes a set of background operations that are reflected by the sequential appearances of toast items in the bottom-right corner. "Applying inventory edits...", "Inventory updated for 1 entry", "Regenerating courses of action...", and then later, a "Coruses of action updated" toast. 

"We're seeing that the course of action is updated. Web obviously need to supply them, so we can go over to the LOGSYNC matrix."

This will be our next slice, but first let's talk about what we've learned from this last slice of the demo.

```


IMAGE 1
![[Pasted image 20260517173815.png]]
![[Pasted image 20260517174033.png]]
See that there's some toasts in the lower right corner: "Applying inventory edits...", "Inventory updated for 1 entry." (green), "Regenerating Courses of Action"

![[Pasted image 20260517174045.png]]
After, see the MRE supply statuses are blacked out in the future too.

![[Pasted image 20260517174151.png]]
"Courses of Action updated" toast


_________


```


```

