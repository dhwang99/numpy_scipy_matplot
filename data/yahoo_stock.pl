#!/usr/bin/perl
#
# Script to download a table of daily stock data in .csv format
# from yahoo.com
#
# Syntax: yahoo.pl <symbol> <startdate> <stopdate>
#   where startdate and stopdate can be in almost any format
#

use Date::Manip;

($symbol,$startdate,$stopdate)=@ARGV;
$startdate = &ParseDate($startdate);
$stopdate = &ParseDate($stopdate);
#print "symbol=$symbol start=$startdate stop=$stopdate\n";

$startday = &UnixDate($startdate,"%d");
$startmon = &UnixDate($startdate,"%m");
$startyear = &UnixDate($startdate,"%y");
$stopday = &UnixDate($stopdate,"%d");
$stopmon = &UnixDate($stopdate,"%m");
$stopyear = &UnixDate($stopdate,"%y");

$port=80; # HTTP
$dataserver="chart.yahoo.com";

$AF_INET=2;
$SOCK_STREAM=1;

$sockaddr='S n a4 x8';

($name,$aliases,$proto)=getprotobyname('tcp');
($name,$aliases,$type,$len,$remoteaddr)=gethostbyname($dataserver);

$remote=pack($sockaddr,$AF_INET,$port,$remoteaddr);

($a,$b,$c,$d)=unpack('C4',$remoteaddr);
#print "remoteaddr=$a.$b.$c.$d port=$port\n";

if (socket(SOCK,$AF_INET,$SOCK_STREAM,$proto)) {
  #print "Socket OK\n";
} else {
  die $!;
}

if (connect(SOCK,$remote)) {
  #print "connect OK\n";
} else {
  die $!;
}

$local=getsockname(SOCK);
($family,$port,$localaddr)=unpack($sockaddr,$local);
($a,$b,$c,$d)=unpack('C4',$localaddr);
#print "remoteaddr=$a.$b.$c.$d port=$port\n";

select(SOCK); $|=1; select(STDOUT);

print SOCK "GET /table.csv?s=$symbol&a=$startmon&b=$startday&c=$startyear&d=$stopmon&e=$stopday&f=$stopyear&g=d&q=q&y=0&z=$symbol&x=.csv HTTP-1.0\n\n";

$start = 0;

while (<SOCK>) {
  if ($start) {
    ($date,$open,$high,$low,$close,$volume) = split(/,/);
    ($day,$month,$year) = split('-',$date);
    $year = int $year;
    if ($year < 80) {
      $year = $year + 2000;
    }
    $dateint = &ParseDate("$month $day, $year");
    $date = &UnixDate($dateint,"%Y/%m/%d");
    print $date,' ',$open,' ',$high,' ',$low,' ',$close,' ',$volume;
  }
  /^Date/ && ($start = 1);
}
