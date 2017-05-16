DATADIR=post-correction/task1
SCRIPTS=scripts
RESULTS=results

%-low-experiment:$(DATADIR)/%-train-low $(DATADIR)/%-train-medium $(DATADIR)/%-train-high $(DATADIR)/%-dev $(DATADIR)/%-test
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.1 $(RESULTS)/$*-test.low.1.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.1 $(RESULTS)/$*-test.low.2.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.1 $(RESULTS)/$*-test.low.3.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.1 $(RESULTS)/$*-test.low.4.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.1 $(RESULTS)/$*-test.low.5.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.1 $(RESULTS)/$*-test.low.6.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.1 $(RESULTS)/$*-test.low.7.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.1 $(RESULTS)/$*-test.low.8.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.1 $(RESULTS)/$*-test.low.9.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.1 $(RESULTS)/$*-test.low.10.sys

	python3 $(SCRIPTS)/vote.py $(RESULTS)/$*-test.low > $(RESULTS)/$*-test.low.sys

%-medium-experiment:$(DATADIR)/%-train-low $(DATADIR)/%-train-medium $(DATADIR)/%-train-high $(DATADIR)/%-dev $(DATADIR)/%-test
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.1.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.2.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.3.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.4.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.5.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.6.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.7.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.8.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.9.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.10.sys

	python3 $(SCRIPTS)/vote.py $(RESULTS)/$*-test.medium > $(RESULTS)/$*-test.medium.sys

%-high-experiment:$(DATADIR)/%-train--low $(DATADIR)/%-train-medium $(DATADIR)/%-train-high $(DATADIR)/%-dev $(DATADIR)/%-test
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.1.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.2.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.3.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.4.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.5.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.6.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.7.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.8.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.9.sys
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.10.sys

	python3 $(SCRIPTS)/vote.py $(RESULTS)/$*-test.high > $(RESULTS)/$*-test.high.sys