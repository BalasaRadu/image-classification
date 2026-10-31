IMAGE=env.sif
DEF_FILE=project.def

.PHONY: all build run status clean

all: build

build: $(IMAGE)

$(IMAGE): $(DEF_FILE)
	apptainer build --ignore-fakeroot-command $(IMAGE) $(DEF_FILE)

run:
	sbatch job.sh $(ARGS)

status:
	squeue -u $(USER)

clean:
	rm -f output_*.txt error_*.txt