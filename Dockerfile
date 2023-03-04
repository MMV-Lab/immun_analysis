FROM condaforge/mambaforge as mamba
COPY environment.yaml .
RUN --mount=type=cache,target=/opt/conda/pkgs mamba env update -p /env --file environment.yaml

RUN find -name '*.a' -delete && \
  rm -rf /env/conda-meta && \
  rm -rf /env/include && \
  rm /env/lib/libpython3.9.so.1.0 && \
  find -name '__pycache__' -type d -exec rm -rf '{}' '+' && \
  rm -rf /env/lib/python3.9/site-packages/pip /env/lib/python3.9/idlelib /env/lib/python3.9/ensurepip \
    /env/lib/libasan.so.5.0.0 \
    /env/lib/libtsan.so.0.0.0 \
    /env/lib/liblsan.so.0.0.0 \
    /env/lib/libubsan.so.1.0.0 \
    /env/bin/x86_64-conda-linux-gnu-ld \
    /env/bin/sqlite3 \
    /env/bin/openssl \
    /env/share/terminfo && \
  #find /env/lib/python3.9/site-packages/scipy -name 'tests' -type d -exec rm -rf '{}' '+' && \
  find /env/lib/python3.9/site-packages/numpy -name 'tests' -type d -exec rm -rf '{}' '+' && \
  #find /env/lib/python3.9/site-packages/matplotlib -name 'tests' -type d -exec rm -rf '{}' '+' && \
  #find /env/lib/python3.9/site-packages/seaborn -name 'tests' -type d -exec rm -rf '{}' '+' && \
  find /env/lib/python3.9/site-packages -name '*.pyx' -delete && \
  rm -rf /env/lib/python3.9/site-packages/uvloop/loop.c

FROM debian:bullseye-slim as deploy
COPY --from=mamba /env /env

ENV PATH /env/bin:$PATH

RUN apt-get update && apt-get install libgl1 -y

COPY . /home/user/immune_analysis/

WORKDIR /home/user/immune_analysis/

CMD [ "snakemake", "-c", "all", "-k" ,"--snakefile", "./Snakemake/Snakefile" ]