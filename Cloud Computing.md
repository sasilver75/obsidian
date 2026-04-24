


![[Pasted image 20260423182935.png]]
Above:
- [[On-Premise]], [[Infrastructure as a Service]] (IaaS), [[Platform as a Service]] (PaaS), [[Software as a Service]] (SaaS))


For [[Remote Sensing|EO]] Data storage:

![[Pasted image 20260423183108.png]]
(Paper from two years ago):
- There's over an exabyte of EO observation data out there.
- It's getting exponentially faster growing, all capturing higher fidelity images at higher file sizes.
- The schematic on the right dteails the entire system of data collection, processing, transfer, compute, storage.
	- In the ground segment: Sometone launches the satellite, but oncei t's in the air, you need a ground station to receive transmission from. You have a mission control center with people there processing the data, making sure there are no issues, doing things like calibration and validation.
		- Once you have this data in a format that's useful, you want people to start using it! So you store it... in some kind of a datacenter.
	- Data center
	- Users don't usually have access to the data center; they usually have to download the data to the their own computer and do local computing, or some data centers will have cloud computing attached, where you don't have write access, but you can still load them quickly and easily and do processing on them. 
	- The entire right half of this is cloud computing. Crucial for most EO segments just due to the total volume.

Why cloud computing?
- Available: Pay for use
- Elastic: Scale up quickly
- Secure (someone else manages software)
- Maintainable (someone else managed hardware)
- Shareable: Access from anywhere

Cons:
- Data security and privacy
- High learning curve 

![[Pasted image 20260423190133.png]]










