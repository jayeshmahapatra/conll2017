DATADIR=all/task1
SCRIPTS=scripts
ENCDEC_RESULTS=encdec_results
ENCDEC_MODELS=encdec_models
EVAL=evaluation
PYTHON=dynetpython

%-encdec-eval:$(ENCDEC_RESULTS)/%-test.low.1.sys
	$(PYTHON) $(EVAL)/evalm.py --gold $(DATADIR)/$*-dev --guess $(ENCDEC_RESULTS)/$*-test.low.sys --task 1

.PRECIOUS:%.model %.sys

$(ENCDEC_MODELS)/%-low.1.model:$(DATADIR)/%-train-low $(DATADIR)/%-train-medium $(DATADIR)/%-train-high $(DATADIR)/%-dev $(DATADIR)/%-test
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev 50 100 0.01 $(ENCDEC_MODELS)/$*-low.1.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev 50 100 0.01 $(ENCDEC_MODELS)/$*-low.2.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev 50 100 0.01 $(ENCDEC_MODELS)/$*-low.3.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev 50 100 0.01 $(ENCDEC_MODELS)/$*-low.4.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev 50 100 0.01 $(ENCDEC_MODELS)/$*-low.5.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev 50 100 0.01 $(ENCDEC_MODELS)/$*-low.6.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev 50 100 0.01 $(ENCDEC_MODELS)/$*-low.7.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev 50 100 0.01 $(ENCDEC_MODELS)/$*-low.8.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev 50 100 0.01 $(ENCDEC_MODELS)/$*-low.9.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-low $(DATADIR)/$*-dev 50 100 0.01 $(ENCDEC_MODELS)/$*-low.10.model

%-encdec-experiment : $(ENCDEC_MODELS)/%-low.1.model $(ENCDEC_MODELS)/%-medium.1.model $(ENCDEC_MODELS)/%-high.1.model
	touch $@

$(ENCDEC_RESULTS)/%-test.low.sys:$(ENCDEC_MODELS)/%-low.1.model
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-low.1.model $(ENCDEC_RESULTS)/$*-test.low.1.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-low.2.model $(ENCDEC_RESULTS)/$*-test.low.2.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-low.3.model $(ENCDEC_RESULTS)/$*-test.low.3.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-low.4.model $(ENCDEC_RESULTS)/$*-test.low.4.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-low.5.model $(ENCDEC_RESULTS)/$*-test.low.5.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-low.6.model $(ENCDEC_RESULTS)/$*-test.low.6.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-low.7.model $(ENCDEC_RESULTS)/$*-test.low.7.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-low.8.model $(ENCDEC_RESULTS)/$*-test.low.8.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-low.9.model $(ENCDEC_RESULTS)/$*-test.low.9.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-low.10.model $(ENCDEC_RESULTS)/$*-test.low.10.sys

	$(PYTHON) $(SCRIPTS)/vote.py $(ENCDEC_RESULTS)/$*-test.low > $(ENCDEC_RESULTS)/$*-test.low.sys

$(ENCDEC_MODELS)/%-medium.1.model:$(DATADIR)/%-train-low $(DATADIR)/%-train-medium $(DATADIR)/%-train-high $(DATADIR)/%-dev $(DATADIR)/%-test
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev 10 100 0.01 $(ENCDEC_MODELS)/$*-medium.1.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev 10 100 0.01 $(ENCDEC_MODELS)/$*-medium.2.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev 10 100 0.01 $(ENCDEC_MODELS)/$*-medium.3.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev 10 100 0.01 $(ENCDEC_MODELS)/$*-medium.4.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev 10 100 0.01 $(ENCDEC_MODELS)/$*-medium.5.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev 10 100 0.01 $(ENCDEC_MODELS)/$*-medium.6.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev 10 100 0.01 $(ENCDEC_MODELS)/$*-medium.7.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev 10 100 0.01 $(ENCDEC_MODELS)/$*-medium.8.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev 10 100 0.01 $(ENCDEC_MODELS)/$*-medium.9.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-medium $(DATADIR)/$*-dev 10 100 0.01 $(ENCDEC_MODELS)/$*-medium.10.model

$(ENCDEC_RESULTS)/%-test.medium.sys:$(ENCDEC_MODELS)/%-medium.1.model
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-medium.1.model $(ENCDEC_RESULTS)/$*-test.medium.1.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-medium.2.model $(ENCDEC_RESULTS)/$*-test.medium.2.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-medium.3.model $(ENCDEC_RESULTS)/$*-test.medium.3.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-medium.4.model $(ENCDEC_RESULTS)/$*-test.medium.4.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-medium.5.model $(ENCDEC_RESULTS)/$*-test.medium.5.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-medium.6.model $(ENCDEC_RESULTS)/$*-test.medium.6.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-medium.7.model $(ENCDEC_RESULTS)/$*-test.medium.7.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-medium.8.model $(ENCDEC_RESULTS)/$*-test.medium.8.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-medium.9.model $(ENCDEC_RESULTS)/$*-test.medium.9.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-medium.10.model $(ENCDEC_RESULTS)/$*-test.medium.10.sys

	$(PYTHON) $(SCRIPTS)/vote.py $(ENCDEC_RESULTS)/$*-test.medium > $(ENCDEC_RESULTS)/$*-test.medium.sys

$(ENCDEC_MODELS)/%-high.1.model:$(DATADIR)/%-train-low $(DATADIR)/%-train-medium $(DATADIR)/%-train-high $(DATADIR)/%-dev $(DATADIR)/%-test
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev 1 100 0.01 $(ENCDEC_MODELS)/$*-high.1.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev 1 100 0.01 $(ENCDEC_MODELS)/$*-high.2.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev 1 100 0.01 $(ENCDEC_MODELS)/$*-high.3.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev 1 100 0.01 $(ENCDEC_MODELS)/$*-high.4.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev 1 100 0.01 $(ENCDEC_MODELS)/$*-high.5.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev 1 100 0.01 $(ENCDEC_MODELS)/$*-high.6.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev 1 100 0.01 $(ENCDEC_MODELS)/$*-high.7.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev 1 100 0.01 $(ENCDEC_MODELS)/$*-high.8.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev 1 100 0.01 $(ENCDEC_MODELS)/$*-high.9.model
	$(PYTHON) $(SCRIPTS)/attention_train.py $(DATADIR)/$*-train-high $(DATADIR)/$*-dev 1 100 0.01 $(ENCDEC_MODELS)/$*-high.10.model

$(ENCDEC_RESULTS)/%-test.high.sys:$(ENCDEC_MODELS)/%-high.1.model
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-high.1.model $(ENCDEC_RESULTS)/$*-test.high.1.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-high.2.model $(ENCDEC_RESULTS)/$*-test.high.2.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-high.3.model $(ENCDEC_RESULTS)/$*-test.high.3.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-high.4.model $(ENCDEC_RESULTS)/$*-test.high.4.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-high.5.model $(ENCDEC_RESULTS)/$*-test.high.5.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-high.6.model $(ENCDEC_RESULTS)/$*-test.high.6.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-high.7.model $(ENCDEC_RESULTS)/$*-test.high.7.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-high.8.model $(ENCDEC_RESULTS)/$*-test.high.8.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-high.9.model $(ENCDEC_RESULTS)/$*-test.high.9.sys
	$(PYTHON) $(SCRIPTS)/attention_test.py $(DATADIR)/$*-test $(ENCDEC_MODELS)/$*-high.10.model $(ENCDEC_RESULTS)/$*-test.high.10.sys

	$(PYTHON) $(SCRIPTS)/vote.py $(ENCDEC_RESULTS)/$*-test.high > $(ENCDEC_RESULTS)/$*-test.high.sys