#
# Regular cron jobs for the distributed-ranges package
#
0 4	* * *	root	[ -x /usr/bin/distributed-ranges_maintenance ] && /usr/bin/distributed-ranges_maintenance
