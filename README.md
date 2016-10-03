<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Pysmad">Pysmad<a class="anchor-link" href="#Pysmad"></a></h1><p>This is the beginning of a set of tools and scripts used by the <a href="https://journalism.wisc.edu/graduate/research-centers-and-groups/">Social Media and Democracy group at University of Wisconsin-Madison</a>.</p>
<p>It is mostly built off of <a href="https://radimrehurek.com/gensim/">gensim</a> and the <a href="http://www.nltk.org/">NLTK</a>.</p>
<p>LDAModel will process an archive of news articles, run LDA topic modeling and generate word clouds. For a demo:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">LDAModel</span>

<span class="n">LDAModel</span><span class="o">.</span><span class="n">main</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stderr output_text">
<pre>Using gpu device 0: Quadro K620M (CNMeM is enabled with initial size: 95.0% of memory, cuDNN 5103)
/usr/lib/python3.5/site-packages/theano/sandbox/cuda/__init__.py:600: UserWarning: Your cuDNN version is more recent than the one Theano officially supports. If you see any problems, try updating Theano or downgrading cuDNN to version 5.
  warnings.warn(warn)
</pre>
</div>
</div>

<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>Tokenizing took 2.344702959060669s
Filtering took 0.24883556365966797s
Stemming took 3.81022047996521s
Creating dictionary took 0.23653340339660645s
Creating BOW embeddings took 0.14314699172973633s
Training LDA model took 8.082087516784668s
article 253 is empty
article 475 is empty
Sorting articles by topic took 8.089366912841797s
</pre>
</div>
</div>

<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stderr output_text">
<pre>/usr/lib/python3.5/site-packages/PIL/ImageDraw.py:100: UserWarning: setfont() is deprecated. Please set the attribute directly instead.
  &#34;Please set the attribute directly instead.&#34;)
</pre>
</div>
</div>

<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>Generating wordclouds took 4.251720905303955s
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Once run, check out the <code>data/</code> directory to see what LDAModel created.  The word clouds and articles sorted by topics are found in <code>data/output</code>.</p>
<p>More coming soon.</p>

</div>
</div>
</div>
