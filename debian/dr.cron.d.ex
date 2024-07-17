#
# Regular cron jobs for the dr package
#
0 4	* * *	root	[ -x /usr/bin/dr_maintenance ] && /usr/bin/dr_maintenance
