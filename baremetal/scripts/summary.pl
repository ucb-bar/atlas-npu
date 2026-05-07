#!/usr/bin/env perl
# Print pass/fail summary for the requested Atlas baremetal tests by reading
# chipyard's per-test logs. Replaces the inline perl one-liner that couldn't
# detect tests dying before reaching the C harness's *** FAILED *** print
# (e.g. SystemVerilog assertion failures, VCS fatals, sim aborts).
#
# Required env:
#   TESTS           — space-separated list of test names
#   SIM_OUTPUT_DIR  — chipyard's output dir for this CONFIG
# Exit code: 0 only if every test PASSED.

use strict;
use warnings;

my $tests      = $ENV{TESTS}          // die "summary.pl: TESTS env var not set\n";
my $output_dir = $ENV{SIM_OUTPUT_DIR} // die "summary.pl: SIM_OUTPUT_DIR not set\n";

my @names = split /\s+/, $tests;
my ($passed, $failed) = (0, 0);
my @rows;

# Verdict is PASSED only when the log explicitly contains *** PASSED ***
# AND no *** FAILED *** appears earlier. Anything else is FAILED — the
# reason text differentiates: explicit C-harness fail marker, SV
# assertion / VCS Fatal, sim ran without verdict, or log missing entirely.
sub classify {
    my $log = shift;
    return ('FAILED', '(no log produced)') unless -f $log;
    open(my $fh, '<', $log) or return ('FAILED', "(can't read log: $!)");

    my $died_reason;
    while (my $line = <$fh>) {
        if ($line =~ /\*{3} FAILED \*{3}\s*(.*)/) {
            close $fh;
            return ('FAILED', $1);
        }
        if ($line =~ /\*{3} PASSED \*{3}\s*(.*)/) {
            close $fh;
            return ('PASSED', $1);
        }
        if (!defined($died_reason)
            && $line =~ /(Assertion failed:.*|^Fatal:.*|FAILED \(\*+\).*)/) {
            chomp(my $r = $1);
            $died_reason = substr($r, 0, 80);
        }
    }
    close $fh;
    return ('FAILED', $died_reason // '(sim ran without PASS/FAIL marker)');
}

for my $name (@names) {
    my $log = "$output_dir/atlas_$name.log";
    my ($verdict, $reason) = classify($log);
    $reason //= '';
    $reason =~ s/^\s+//;

    if ($verdict eq 'PASSED') { $passed++; } else { $failed++; }
    push @rows, sprintf("  [%s] %-45s %s", $verdict, $name, $reason);
}

my $total = scalar @names;

print "\n";
print "================ Test Summary ($total tests) ================\n";
print "$_\n" for @rows;
print "============================================================\n";
if ($failed == 0) {
    print "All $total tests passed.\n";
} else {
    print "$passed / $total passed, $failed failed.\n";
}
print "\n";

exit($failed == 0 ? 0 : 1);
