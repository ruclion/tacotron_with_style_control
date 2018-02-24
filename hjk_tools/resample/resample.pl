#!/usr/local/bin/perl


if ($#ARGV != 2) {
	print "Converting wave files to raw (NOHEAD) files\n";
	print "Usage: $0 sox_exe_file src_wave_path out_wave_path\n";
	exit(1);
}

$soxExeFile = $ARGV[0];
$inWavePath = $ARGV[1];
$outRawPath = $ARGV[2];

$inWavePath =~ s/\/$//;
$outRawPath =~ s/\/$//;

opendir(DIR, $inWavePath) || die "cannot open $inWavePath: $! \n";
system "mkdir -p $outRawPath";

print "Resampling wave files ...\n";

while ($waveFile = readdir(DIR)) {

	if ($waveFile =~ /\.wav/i) {
		# Convert to lower case
		$rawFile = $waveFile;
		#$rawFile =~ s/\.wav/\.raw/i;
		# Executing now
		#$cmd = "$soxExeFile $inWavePath/$waveFile $outRawPath/$rawFile";
		$cmd = "$soxExeFile $inWavePath/$waveFile $outRawPath/$waveFile.tmp.wav rate -s -a 16000 dither -s ";
		system $cmd;
		#$cmd = "$soxExeFile $outRawPath/$waveFile.tmp.wav $outRawPath/$rawFile remix 1";
		#system $cmd;
		print "$cmd\n";
		#$cmd = "rm $outRawPath/$waveFile.tmp.wav";
		#system $cmd;
	}
}

closedir DIR;

print "Done!\n";
