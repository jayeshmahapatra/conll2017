DATADIR=post-correction/task1
SCRIPTS=scripts
RESULTS=results
MODELS=models

%-low-experiment:$(DATADIR)/%-train-low $(DATADIR)/%-train-medium $(DATADIR)/%-train-high $(DATADIR)/%-dev $(DATADIR)/%-test
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.01 $(RESULTS)/$*-test.low.1.sys $(MODELS)/$*-test.low-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.01 $(RESULTS)/$*-test.low.2.sys $(MODELS)/$*-test.low-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.01 $(RESULTS)/$*-test.low.3.sys $(MODELS)/$*-test.low-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.01 $(RESULTS)/$*-test.low.4.sys $(MODELS)/$*-test.low-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.01 $(RESULTS)/$*-test.low.5.sys $(MODELS)/$*-test.low-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.01 $(RESULTS)/$*-test.low.6.sys $(MODELS)/$*-test.low-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.01 $(RESULTS)/$*-test.low.7.sys $(MODELS)/$*-test.low-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.01 $(RESULTS)/$*-test.low.8.sys $(MODELS)/$*-test.low-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.01 $(RESULTS)/$*-test.low.9.sys $(MODELS)/$*-test.low-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev $(DATADIR)/$*-test 50 100 0.01 $(RESULTS)/$*-test.low.10.sys $(MODELS)/$*-test.low-post-correction

	python3 $(SCRIPTS)/vote.py $(RESULTS)/$*-test.low > $(RESULTS)/$*-test.low.sys

%-medium-experiment:$(DATADIR)/%-train-low $(DATADIR)/%-train-medium $(DATADIR)/%-train-high $(DATADIR)/%-dev $(DATADIR)/%-test
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.1.sys $(MODELS)/$*-test.medium-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.2.sys $(MODELS)/$*-test.medium-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.3.sys $(MODELS)/$*-test.medium-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.4.sys $(MODELS)/$*-test.medium-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.5.sys $(MODELS)/$*-test.medium-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.6.sys $(MODELS)/$*-test.medium-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.7.sys $(MODELS)/$*-test.medium-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.8.sys $(MODELS)/$*-test.medium-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.9.sys $(MODELS)/$*-test.medium-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev $(DATADIR)/$*-test 10 100 0.01 $(RESULTS)/$*-test.medium.10.sys $(MODELS)/$*-test.medium-post-correction

	python3 $(SCRIPTS)/vote.py $(RESULTS)/$*-test.medium > $(RESULTS)/$*-test.medium.sys

%-high-experiment:$(DATADIR)/%-train--low $(DATADIR)/%-train-medium $(DATADIR)/%-train-high $(DATADIR)/%-dev $(DATADIR)/%-test
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.1.sys $(RESULTS)/$*-test.high-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.2.sys $(RESULTS)/$*-test.high-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.3.sys $(RESULTS)/$*-test.high-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.4.sys $(RESULTS)/$*-test.high-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.5.sys $(RESULTS)/$*-test.high-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.6.sys $(RESULTS)/$*-test.high-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.7.sys $(RESULTS)/$*-test.high-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.8.sys $(RESULTS)/$*-test.high-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.9.sys $(RESULTS)/$*-test.high-post-correction
	python3 $(SCRIPTS)/post_correction_attention.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev $(DATADIR)/$*-test 1 100 0.01 $(RESULTS)/$*-test.high.10.sys $(RESULTS)/$*-test.high-post-correction

	python3 $(SCRIPTS)/vote.py $(RESULTS)/$*-test.high > $(RESULTS)/$*-test.high.sys