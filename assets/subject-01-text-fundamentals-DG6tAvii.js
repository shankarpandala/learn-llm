import{j as e,r as _}from"./vendor-DWbzdFaj.js";import{r}from"./vendor-katex-BYl39Yo6.js";import{A as E,m as $}from"./vendor-motion-Bf53smr3.js";function N(n){return n?n.split(/(\$[^$]+\$)/g).map((t,s)=>{if(t.startsWith("$")&&t.endsWith("$")){const h=t.slice(1,-1);return e.jsx(r.InlineMath,{math:h},s)}return e.jsx("span",{children:t},s)}):null}function c({title:n,children:a,definition:t,notation:s,id:h}){return e.jsxs("div",{id:h,className:"my-6 rounded-xl border border-purple-200 bg-purple-50/60 dark:border-purple-500/30 dark:bg-purple-950/30",children:[e.jsxs("div",{className:"flex items-start gap-3 px-5 pt-4 pb-2",children:[e.jsx("span",{className:"flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-purple-600 text-xs font-bold text-white","aria-hidden":"true",children:"D"}),e.jsx("h4",{className:"text-base font-semibold text-purple-900 dark:text-purple-200 leading-snug pt-0.5",children:N(n)})]}),e.jsxs("div",{className:"px-5 pb-4 pl-[3.25rem] text-sm leading-relaxed text-gray-700 dark:text-gray-300",children:[t?e.jsx("p",{children:N(t)}):a,s&&e.jsxs("p",{className:"mt-3 text-sm italic text-purple-700 dark:text-purple-400",children:[e.jsx("span",{className:"font-medium",children:"Notation: "}),N(s)]})]})]})}function v(n){return n?n.split(/(\$[^$]+\$)/g).map((t,s)=>t.startsWith("$")&&t.endsWith("$")?e.jsx(r.InlineMath,{math:t.slice(1,-1)},s):e.jsx("span",{children:t},s)):null}function p({title:n,problem:a,steps:t,children:s,id:h}){const[i,m]=_.useState(!1);return e.jsxs("div",{id:h,className:"my-6 rounded-xl border border-emerald-200 bg-emerald-50/60 dark:border-emerald-500/30 dark:bg-emerald-950/30",children:[e.jsxs("div",{className:"flex items-start gap-3 px-5 pt-4 pb-2",children:[e.jsx("span",{className:"flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-emerald-600 text-xs font-bold text-white","aria-hidden":"true",children:"E"}),e.jsx("h4",{className:"text-base font-semibold text-emerald-900 dark:text-emerald-200 leading-snug pt-0.5",children:v(n)})]}),e.jsx("div",{className:"px-5 pb-3 pl-[3.25rem] text-sm leading-relaxed text-gray-700 dark:text-gray-300",children:a?e.jsx("p",{children:v(a)}):s}),t&&t.length>0&&e.jsxs("div",{className:"border-t border-emerald-200 dark:border-emerald-500/20",children:[e.jsxs("button",{onClick:()=>m(!i),className:"flex w-full items-center gap-2 px-5 py-3 text-sm font-medium text-emerald-700 dark:text-emerald-300 hover:bg-emerald-100/60 dark:hover:bg-emerald-900/30 transition-colors","aria-expanded":i,children:[e.jsx("svg",{className:`h-4 w-4 shrink-0 transition-transform ${i?"rotate-90":""}`,fill:"none",viewBox:"0 0 24 24",stroke:"currentColor",strokeWidth:2,children:e.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M9 5l7 7-7 7"})}),i?"Hide Solution":"Show Solution"]}),e.jsx(E,{children:i&&e.jsx($.div,{initial:{height:0,opacity:0},animate:{height:"auto",opacity:1},exit:{height:0,opacity:0},transition:{duration:.25},className:"overflow-hidden",children:e.jsx("div",{className:"px-5 pb-4 pl-[3.25rem]",children:e.jsx("ol",{className:"space-y-4",children:t.map((o,x)=>e.jsxs("li",{className:"relative pl-8",children:[e.jsx("span",{className:"absolute left-0 top-0 flex h-6 w-6 items-center justify-center rounded-full bg-emerald-600 text-[11px] font-bold text-white",children:x+1}),e.jsxs("div",{className:"flex flex-col gap-1",children:[o.formula&&e.jsx("span",{className:"font-mono text-sm text-emerald-800 dark:text-emerald-200",children:v(o.formula)}),o.explanation&&e.jsx("span",{className:"text-sm text-gray-600 dark:text-gray-400",children:v(o.explanation)})]})]},x))})})})})]})]})}function z(n){return n?n.split(/(\$[^$]+\$)/g).map((t,s)=>t.startsWith("$")&&t.endsWith("$")?e.jsx(r.InlineMath,{math:t.slice(1,-1)},s):e.jsx("span",{children:t},s)):null}const M={note:{border:"border-blue-200 dark:border-blue-500/30",bg:"bg-blue-50/60 dark:bg-blue-950/30",iconColor:"text-blue-500 dark:text-blue-400",titleColor:"text-blue-900 dark:text-blue-200",icon:e.jsx("svg",{className:"h-5 w-5",fill:"none",viewBox:"0 0 24 24",stroke:"currentColor",strokeWidth:2,children:e.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"})})},historical:{border:"border-amber-200 dark:border-amber-500/30",bg:"bg-amber-50/60 dark:bg-amber-950/30",iconColor:"text-amber-500 dark:text-amber-400",titleColor:"text-amber-900 dark:text-amber-200",icon:e.jsx("svg",{className:"h-5 w-5",fill:"none",viewBox:"0 0 24 24",stroke:"currentColor",strokeWidth:2,children:e.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"})})},intuition:{border:"border-violet-200 dark:border-violet-500/30",bg:"bg-violet-50/60 dark:bg-violet-950/30",iconColor:"text-violet-500 dark:text-violet-400",titleColor:"text-violet-900 dark:text-violet-200",icon:e.jsx("svg",{className:"h-5 w-5",fill:"none",viewBox:"0 0 24 24",stroke:"currentColor",strokeWidth:2,children:e.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"})})},tip:{border:"border-teal-200 dark:border-teal-500/30",bg:"bg-teal-50/60 dark:bg-teal-950/30",iconColor:"text-teal-500 dark:text-teal-400",titleColor:"text-teal-900 dark:text-teal-200",icon:e.jsx("svg",{className:"h-5 w-5",fill:"none",viewBox:"0 0 24 24",stroke:"currentColor",strokeWidth:2,children:e.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M5 13l4 4L19 7"})})}};function l({type:n="note",title:a,content:t,children:s,id:h}){const i=M[n]||M.note;return e.jsx("div",{id:h,className:`my-6 rounded-xl border ${i.border} ${i.bg}`,children:e.jsxs("div",{className:"flex items-start gap-3 px-5 py-4",children:[e.jsx("span",{className:`shrink-0 mt-0.5 ${i.iconColor}`,"aria-hidden":"true",children:i.icon}),e.jsxs("div",{className:"min-w-0 flex-1",children:[a&&e.jsx("h4",{className:`text-sm font-semibold ${i.titleColor} mb-1`,children:z(a)}),e.jsx("div",{className:"text-sm leading-relaxed text-gray-700 dark:text-gray-300",children:t?e.jsx("p",{children:z(t)}):s})]})]})})}function S(n){return n?n.split(/(\$[^$]+\$)/g).map((t,s)=>t.startsWith("$")&&t.endsWith("$")?e.jsx(r.InlineMath,{math:t.slice(1,-1)},s):e.jsx("span",{children:t},s)):null}function u({title:n="Warning",content:a,children:t,id:s}){return e.jsx("div",{id:s,className:"my-6 rounded-xl border border-amber-300 bg-amber-50/70 dark:border-amber-500/30 dark:bg-amber-950/30",children:e.jsxs("div",{className:"flex items-start gap-3 px-5 py-4",children:[e.jsx("span",{className:"shrink-0 mt-0.5 text-amber-500 dark:text-amber-400","aria-hidden":"true",children:e.jsx("svg",{className:"h-5 w-5",fill:"none",viewBox:"0 0 24 24",stroke:"currentColor",strokeWidth:2,children:e.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"})})}),e.jsxs("div",{className:"min-w-0 flex-1",children:[n&&e.jsx("h4",{className:"text-sm font-semibold text-amber-900 dark:text-amber-200 mb-1",children:S(n)}),e.jsx("div",{className:"text-sm leading-relaxed text-gray-700 dark:text-gray-300",children:a?e.jsx("p",{children:S(a)}):t})]})]})})}function L(n){const a=[];let t=0;const s=new Set(["False","None","True","and","as","assert","async","await","break","class","continue","def","del","elif","else","except","finally","for","from","global","if","import","in","is","lambda","nonlocal","not","or","pass","raise","return","try","while","with","yield"]),h=new Set(["print","range","len","int","float","str","list","dict","set","tuple","bool","type","isinstance","enumerate","zip","map","filter","sorted","reversed","min","max","sum","abs","any","all","open","input","super","property","staticmethod","classmethod","hasattr","getattr","setattr","delattr","callable","iter","next","repr","format","id","hash","hex","oct","bin","ord","chr","round","pow","divmod","vars","dir","help","ValueError","TypeError","KeyError","IndexError","RuntimeError","StopIteration","Exception","NotImplementedError","AttributeError","ImportError","FileNotFoundError","ZeroDivisionError","OSError"]);for(;t<n.length;){const i=n[t];if(i===" "||i==="	"){let m=t;for(;t<n.length&&(n[t]===" "||n[t]==="	");)t++;a.push({type:"plain",value:n.slice(m,t)});continue}if(i===`
`){a.push({type:"plain",value:`
`}),t++;continue}if(i==="#"){let m=t;for(;t<n.length&&n[t]!==`
`;)t++;a.push({type:"comment",value:n.slice(m,t)});continue}if((i==='"'||i==="'")&&n.slice(t,t+3)===i.repeat(3)){const m=i.repeat(3);let o=t;for(t+=3;t<n.length&&n.slice(t,t+3)!==m;)n[t]==="\\"&&t++,t++;t+=3,a.push({type:"string",value:n.slice(o,t)});continue}if(i==='"'||i==="'"){let m=t;const o=i;for(t++;t<n.length&&n[t]!==o&&n[t]!==`
`;)n[t]==="\\"&&t++,t++;t<n.length&&n[t]===o&&t++,a.push({type:"string",value:n.slice(m,t)});continue}if((i==="f"||i==="F")&&t+1<n.length&&(n[t+1]==='"'||n[t+1]==="'")){let m=t;t++;const o=n[t];for(t++;t<n.length&&n[t]!==o&&n[t]!==`
`;)n[t]==="\\"&&t++,t++;t<n.length&&n[t]===o&&t++,a.push({type:"string",value:n.slice(m,t)});continue}if(/\d/.test(i)||i==="."&&t+1<n.length&&/\d/.test(n[t+1])){let m=t;if(i==="0"&&t+1<n.length&&(n[t+1]==="x"||n[t+1]==="X"))for(t+=2;t<n.length&&/[0-9a-fA-F_]/.test(n[t]);)t++;else{for(;t<n.length&&/[\d_]/.test(n[t]);)t++;if(t<n.length&&n[t]===".")for(t++;t<n.length&&/[\d_]/.test(n[t]);)t++;if(t<n.length&&(n[t]==="e"||n[t]==="E"))for(t++,t<n.length&&(n[t]==="+"||n[t]==="-")&&t++;t<n.length&&/[\d_]/.test(n[t]);)t++}t<n.length&&(n[t]==="j"||n[t]==="J")&&t++,a.push({type:"number",value:n.slice(m,t)});continue}if(/[a-zA-Z_]/.test(i)){let m=t;for(;t<n.length&&/[a-zA-Z0-9_]/.test(n[t]);)t++;const o=n.slice(m,t);s.has(o)?a.push({type:"keyword",value:o}):h.has(o)?a.push({type:"builtin",value:o}):o==="self"||o==="cls"?a.push({type:"self",value:o}):a.push({type:"identifier",value:o});continue}if(i==="@"){let m=t;for(t++;t<n.length&&/[a-zA-Z0-9_.]/.test(n[t]);)t++;a.push({type:"decorator",value:n.slice(m,t)});continue}a.push({type:"operator",value:i}),t++}return a}const C={keyword:"text-purple-400",builtin:"text-cyan-400",string:"text-green-400",number:"text-orange-400",comment:"text-gray-500 italic",decorator:"text-yellow-400",self:"text-red-400",operator:"text-gray-400",identifier:"text-gray-200",plain:""};function d({code:n="",title:a,colabUrl:t,showLines:s=!0,id:h}){const[i,m]=_.useState(!1),o=_.useCallback(()=>{navigator.clipboard.writeText(n).then(()=>{m(!0),setTimeout(()=>m(!1),2e3)})},[n]),x=L(n),g=n.split(`
`);g.length;const f=x.map((w,k)=>{const b=C[w.type];return b?e.jsx("span",{className:b,children:w.value},k):e.jsx("span",{children:w.value},k)});return e.jsxs("div",{id:h,className:"my-6 overflow-hidden rounded-xl border border-gray-700 bg-gray-900 shadow-sm",children:[e.jsxs("div",{className:"flex items-center justify-between border-b border-gray-700 bg-gray-800/80 px-4 py-2",children:[e.jsxs("div",{className:"flex items-center gap-3",children:[e.jsxs("div",{className:"flex items-center gap-1.5","aria-hidden":"true",children:[e.jsx("span",{className:"h-3 w-3 rounded-full bg-red-500/80"}),e.jsx("span",{className:"h-3 w-3 rounded-full bg-yellow-500/80"}),e.jsx("span",{className:"h-3 w-3 rounded-full bg-green-500/80"})]}),a&&e.jsx("span",{className:"text-xs font-medium text-gray-400",children:a})]}),e.jsxs("div",{className:"flex items-center gap-2",children:[e.jsx("span",{className:"rounded bg-gray-700/60 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-gray-400",children:"Python"}),t&&e.jsxs("a",{href:t,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 rounded bg-yellow-600/20 px-2 py-0.5 text-[10px] font-semibold text-yellow-400 hover:bg-yellow-600/30 transition-colors",children:[e.jsx("svg",{className:"h-3 w-3",viewBox:"0 0 24 24",fill:"currentColor",children:e.jsx("path",{d:"M16.9414 4.9757a7.033 7.033 0 0 0-9.8728 1.7054l3.6068 2.5708a3.52 3.52 0 0 1 4.9392-.8525 3.521 3.521 0 0 1 .8525 4.9392l3.6068 2.5709a7.033 7.033 0 0 0-.3325-8.8338zm-9.1057 4.1884L4.229 6.5933a7.033 7.033 0 0 0 .3325 8.8338 7.033 7.033 0 0 0 9.8728-1.7054l-3.6068-2.5708a3.52 3.52 0 0 1-4.9392.8524 3.52 3.52 0 0 1-.8525-4.939z"})}),"Colab"]}),e.jsx("button",{onClick:o,className:"flex items-center gap-1 rounded bg-gray-700/60 px-2 py-0.5 text-[10px] font-semibold text-gray-400 hover:bg-gray-600/60 hover:text-gray-300 transition-colors","aria-label":"Copy code",children:i?e.jsxs(e.Fragment,{children:[e.jsx("svg",{className:"h-3 w-3",fill:"none",viewBox:"0 0 24 24",stroke:"currentColor",strokeWidth:2,children:e.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M5 13l4 4L19 7"})}),"Copied"]}):e.jsxs(e.Fragment,{children:[e.jsx("svg",{className:"h-3 w-3",fill:"none",viewBox:"0 0 24 24",stroke:"currentColor",strokeWidth:2,children:e.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"})}),"Copy"]})})]})]}),e.jsx("div",{className:"overflow-x-auto",children:e.jsx("pre",{className:"p-4 text-sm leading-relaxed font-mono",children:s?e.jsx("table",{className:"w-full border-collapse",children:e.jsx("tbody",{children:g.map((w,k)=>e.jsxs("tr",{className:"hover:bg-gray-800/50",children:[e.jsx("td",{className:"select-none pr-4 text-right align-top text-xs text-gray-600 w-8",children:k+1}),e.jsx("td",{className:"whitespace-pre text-gray-200",children:L(w).map((b,T)=>{const P=C[b.type];return P?e.jsx("span",{className:P,children:b.value},T):e.jsx("span",{children:b.value},T)})})]},k))})}):e.jsx("code",{children:f})})})]})}function B(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Tokenization: Word, Subword, and Character"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Tokenization is the first step in any NLP pipeline. It converts raw text into discrete units (tokens) that a model can process. The choice of tokenization strategy profoundly affects vocabulary size, the ability to handle rare words, and overall model performance."}),e.jsx(c,{title:"Tokenization",definition:"Tokenization is the process of splitting a sequence of text into smaller units called tokens. Tokens may be words, subwords, or individual characters depending on the strategy used.",id:"def-tokenization"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Word Tokenization"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The simplest approach splits text on whitespace and punctuation. While intuitive, word-level tokenization creates enormous vocabularies and cannot handle out-of-vocabulary (OOV) words."}),e.jsx(d,{title:"word_tokenization.py",code:`# Simple word tokenization approaches
text = "The transformer architecture hasn't been surpassed yet."

# Naive whitespace split
tokens_naive = text.split()
print("Whitespace:", tokens_naive)
# ['The', 'transformer', 'architecture', "hasn't", 'been', 'surpassed', 'yet.']

# NLTK word tokenizer handles punctuation and contractions
import nltk
tokens_nltk = nltk.word_tokenize(text)
print("NLTK:", tokens_nltk)
# ['The', 'transformer', 'architecture', 'has', "n't", 'been', 'surpassed', 'yet', '.']

# Problem: OOV words in a fixed vocabulary
vocab = set(tokens_nltk)
new_text = "Transformers revolutionized NLP"
new_tokens = nltk.word_tokenize(new_text)
oov = [t for t in new_tokens if t not in vocab]
print(f"Out-of-vocabulary: {oov}")  # All tokens are OOV`,id:"code-word-tokenization"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Byte Pair Encoding (BPE)"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"BPE starts with individual characters and iteratively merges the most frequent adjacent pair of tokens. This produces a subword vocabulary that balances between character-level and word-level granularity. GPT-2, GPT-3, and GPT-4 all use variants of BPE."}),e.jsx(p,{title:"BPE Merge Example",problem:"Given the corpus with word frequencies: {'low': 5, 'lower': 2, 'newest': 6, 'widest': 3}, show the first BPE merge.",steps:[{formula:"Initial vocabulary: {l, o, w, e, r, n, s, t, i, d}",explanation:"Start with all individual characters."},{formula:"Count pairs: (e, s) appears 9 times (6+3)",explanation:'The pair "e"+"s" is the most frequent across "newest" and "widest".'},{formula:'Merge: es -> new token "es"',explanation:'Replace all occurrences of "e" followed by "s" with the merged token "es".'},{formula:"Next: (es, t) appears 9 times",explanation:"Continue merging the most frequent pair in the updated corpus."}],id:"example-bpe"}),e.jsx(d,{title:"bpe_with_tiktoken.py",code:`# Using tiktoken (OpenAI's fast BPE tokenizer)
import tiktoken

# GPT-4 uses cl100k_base encoding
enc = tiktoken.get_encoding("cl100k_base")

text = "Tokenization is fundamental to LLMs."
tokens = enc.encode(text)
print(f"Token IDs: {tokens}")
print(f"Number of tokens: {len(tokens)}")

# Decode individual tokens to see subwords
for tid in tokens:
    print(f"  {tid} -> '{enc.decode([tid])}'")

# Compare token counts for different texts
examples = [
    "Hello world",
    "antidisestablishmentarianism",  # Long word gets split
    "こんにちは世界",  # Non-English text
]
for ex in examples:
    toks = enc.encode(ex)
    print(f"'{ex}' -> {len(toks)} tokens")`,id:"code-bpe"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"WordPiece and SentencePiece"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"WordPiece (used by BERT) is similar to BPE but selects merges based on likelihood rather than frequency. SentencePiece treats the input as a raw byte stream, making it language-agnostic and able to handle any Unicode text without pre-tokenization."}),e.jsx(l,{type:"intuition",title:"Why Subword Tokenization Works",content:"Subword methods capture morphological structure. The word 'unhappiness' might be split into 'un', 'happiness' or 'un', 'happy', 'ness'. This lets models generalize: if they know 'happy' and see 'un' + 'happy', they can infer meaning compositionally.",id:"note-subword-intuition"}),e.jsx(u,{title:"Tokenization Affects Everything Downstream",content:"The choice of tokenizer determines the model's effective context length. A sentence that takes 10 tokens with one tokenizer might take 20 with another. This is why token counts, not word counts, determine cost and context limits in LLM APIs.",id:"warning-tokenization"}),e.jsx(l,{type:"historical",title:"Evolution of Tokenization",content:"Early NLP used word-level tokens (Word2Vec, 2013). BPE was adapted from data compression to NLP by Sennrich et al. (2016). Google introduced WordPiece for BERT (2018). SentencePiece (Kudo & Richardson, 2018) unified subword tokenization into a language-independent framework used by T5, LLaMA, and many modern models.",id:"note-history"})]})}const ee=Object.freeze(Object.defineProperty({__proto__:null,default:B},Symbol.toStringTag,{value:"Module"}));function q(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Unicode and Text Encoding"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Before text reaches a tokenizer, it exists as a sequence of bytes governed by encoding standards. Understanding Unicode and UTF-8 is essential for building robust NLP systems that handle multilingual text, emojis, and special characters."}),e.jsx(c,{title:"Unicode Code Point",definition:"A Unicode code point is a unique integer assigned to every character in the Unicode standard. Code points are written as U+XXXX (hexadecimal). Unicode 15.0 defines over 149,000 characters across 161 scripts.",notation:"A character like 'A' has code point $U{+}0041$, stored as integer $65_{10}$.",id:"def-unicode"}),e.jsx(c,{title:"UTF-8 Encoding",definition:"UTF-8 is a variable-length encoding that represents each Unicode code point as 1 to 4 bytes. ASCII characters (U+0000 to U+007F) use 1 byte, making UTF-8 backward-compatible with ASCII.",id:"def-utf8"}),e.jsx(p,{title:"UTF-8 Byte Sequences",problem:"How is the emoji character (U+1F600) encoded in UTF-8?",steps:[{formula:"Code point: U+1F600 = 128512 in decimal",explanation:"The grinning face emoji has a high code point requiring 4 bytes."},{formula:"Binary: 0001 1111 0110 0000 0000",explanation:"Convert the code point to binary (21 bits needed)."},{formula:"UTF-8 pattern: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx",explanation:"4-byte UTF-8 uses this bit pattern (leading byte starts with 11110)."},{formula:"Result: F0 9F 98 80 (4 bytes)",explanation:"Fill the x positions with the binary bits of the code point."}],id:"example-utf8"}),e.jsx(d,{title:"unicode_exploration.py",code:`# Exploring Unicode and encodings in Python
text = "Hello, 世界! 😀"

# Code points
for char in text:
    print(f"'{char}' -> U+{ord(char):04X} (decimal {ord(char)})")

# UTF-8 encoding
utf8_bytes = text.encode('utf-8')
print(f"\\nUTF-8 bytes: {utf8_bytes}")
print(f"UTF-8 length: {len(utf8_bytes)} bytes")
print(f"Character count: {len(text)} characters")

# Different encodings produce different byte lengths
encodings = ['utf-8', 'utf-16', 'utf-32', 'ascii']
for enc in encodings:
    try:
        encoded = text.encode(enc)
        print(f"{enc:8s}: {len(encoded)} bytes")
    except UnicodeEncodeError:
        print(f"{enc:8s}: Cannot encode (characters out of range)")

# Byte-level view of a CJK character
char = '世'  # Chinese character for "world"
print(f"\\n'{char}' in UTF-8: {char.encode('utf-8').hex(' ')}")
print(f"'{char}' in UTF-16: {char.encode('utf-16-be').hex(' ')}")`,id:"code-unicode"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Byte-Level Tokenization"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Modern LLMs like GPT-2 and beyond use byte-level BPE, which operates on raw UTF-8 bytes rather than Unicode characters. This guarantees that any input string can be tokenized without unknown tokens, since all 256 byte values are in the base vocabulary."}),e.jsx(d,{title:"byte_level_tokenization.py",code:`# Byte-level BPE: the foundation of GPT tokenizers
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

# Multilingual text: same tokenizer handles everything
examples = {
    "English": "Machine learning is powerful.",
    "Chinese": "机器学习很强大。",
    "Arabic":  "التعلم الآلي",
    "Emoji":   "🤖💬🌍",
}

for lang, text in examples.items():
    tokens = enc.encode(text)
    utf8_len = len(text.encode('utf-8'))
    print(f"{lang:10s}: {len(tokens):3d} tokens, {utf8_len:3d} UTF-8 bytes")
    # Decode each token to see the subwords
    decoded = [enc.decode([t]) for t in tokens]
    print(f"{'':10s}  Tokens: {decoded}")`,id:"code-byte-bpe"}),e.jsx(l,{type:"tip",title:"Normalization Matters",content:"Unicode has multiple ways to represent the same visual character. For example, the accented 'e' can be a single code point (U+00E9) or a base 'e' plus a combining accent (U+0065 + U+0301). Always normalize text (NFC or NFKC) before tokenization to avoid treating identical-looking text differently.",id:"note-normalization"}),e.jsx(u,{title:"Encoding Mismatches Cause Silent Bugs",content:"If you read a file as latin-1 but it was written as UTF-8, multi-byte characters will be split incorrectly. Always be explicit about encodings when reading data: use open(file, encoding='utf-8') in Python.",id:"warning-encoding"}),e.jsx(l,{type:"historical",title:"From ASCII to Unicode",content:"ASCII (1963) defined 128 characters for English. Various incompatible extensions (Latin-1, Shift-JIS, GB2312) arose for other languages. Unicode (1991) unified all scripts into a single standard. UTF-8 (1993), designed by Ken Thompson and Rob Pike, became the dominant encoding on the web, covering over 98% of web pages today.",id:"note-ascii-history"})]})}const te=Object.freeze(Object.defineProperty({__proto__:null,default:q},Symbol.toStringTag,{value:"Module"}));function y(n){return n?n.split(/(\$[^$]+\$)/g).map((t,s)=>t.startsWith("$")&&t.endsWith("$")?e.jsx(r.InlineMath,{math:t.slice(1,-1)},s):e.jsx("span",{children:t},s)):null}function j({title:n,statement:a,proof:t,proofSteps:s,corollaries:h,children:i,id:m}){const[o,x]=_.useState(!1);return e.jsxs("div",{id:m,className:"my-6 rounded-xl border border-indigo-200 bg-indigo-50/60 dark:border-indigo-500/30 dark:bg-indigo-950/30",children:[e.jsxs("div",{className:"flex items-start gap-3 px-5 pt-4 pb-2",children:[e.jsx("span",{className:"flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-indigo-600 text-xs font-bold text-white","aria-hidden":"true",children:"T"}),e.jsx("h4",{className:"text-base font-semibold text-indigo-900 dark:text-indigo-200 leading-snug pt-0.5",children:y(n)})]}),e.jsx("div",{className:"px-5 pb-3 pl-[3.25rem] text-sm leading-relaxed text-gray-700 dark:text-gray-300",children:a?e.jsx("p",{children:y(a)}):i}),h&&h.length>0&&e.jsxs("div",{className:"px-5 pb-3 pl-[3.25rem]",children:[e.jsx("p",{className:"text-xs font-semibold uppercase tracking-wide text-indigo-600 dark:text-indigo-400 mb-1",children:"Corollaries"}),e.jsx("ul",{className:"list-disc list-inside space-y-1 text-sm text-gray-700 dark:text-gray-300",children:h.map((g,f)=>e.jsx("li",{children:y(g)},f))})]}),(t||s)&&e.jsxs("div",{className:"border-t border-indigo-200 dark:border-indigo-500/20",children:[e.jsxs("button",{onClick:()=>x(!o),className:"flex w-full items-center gap-2 px-5 py-3 text-sm font-medium text-indigo-700 dark:text-indigo-300 hover:bg-indigo-100/60 dark:hover:bg-indigo-900/30 transition-colors","aria-expanded":o,children:[e.jsx("svg",{className:`h-4 w-4 shrink-0 transition-transform ${o?"rotate-90":""}`,fill:"none",viewBox:"0 0 24 24",stroke:"currentColor",strokeWidth:2,children:e.jsx("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M9 5l7 7-7 7"})}),o?"Hide Proof":"Show Proof"]}),e.jsx(E,{children:o&&e.jsx($.div,{initial:{height:0,opacity:0},animate:{height:"auto",opacity:1},exit:{height:0,opacity:0},transition:{duration:.25},className:"overflow-hidden",children:e.jsxs("div",{className:"px-5 pb-4 pl-[3.25rem] text-sm leading-relaxed text-gray-700 dark:text-gray-300",children:[t&&e.jsx("p",{children:y(t)}),s&&e.jsx("ol",{className:"space-y-3 mt-2",children:s.map((g,f)=>e.jsxs("li",{className:"flex flex-col gap-1",children:[e.jsxs("span",{className:"text-xs font-semibold text-indigo-500 dark:text-indigo-400",children:["Step ",f+1]}),g.formula&&e.jsx("span",{className:"font-mono text-indigo-800 dark:text-indigo-200",children:y(g.formula)}),g.explanation&&e.jsx("span",{className:"text-gray-600 dark:text-gray-400",children:y(g.explanation)})]},f))}),e.jsx("div",{className:"mt-4 text-right text-lg text-indigo-400 dark:text-indigo-500 select-none",children:"■"})]})})})]})]})}function F(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Bag-of-Words and TF-IDF"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Before neural embeddings, the standard way to represent documents as numerical vectors was through count-based methods. Bag-of-Words and TF-IDF remain important baselines and are still used in information retrieval, search engines, and lightweight classifiers."}),e.jsx(c,{title:"Bag-of-Words (BoW)",definition:"A Bag-of-Words representation converts a document into a vector of word counts (or binary indicators), discarding word order. Each dimension corresponds to a unique word in the vocabulary.",notation:"For a vocabulary of size $V$, each document $d$ becomes a vector $\\mathbf{x} \\in \\mathbb{R}^V$ where $x_i = \\text{count}(w_i, d)$.",id:"def-bow"}),e.jsx(c,{title:"Term Frequency (TF)",definition:"Term frequency measures how often a term $t$ appears in a document $d$. The raw count is often normalized to prevent bias toward longer documents.",id:"def-tf"}),e.jsxs("div",{className:"my-4",children:[e.jsx(r.BlockMath,{math:"\\text{tf}(t, d) = \\frac{f_{t,d}}{\\sum_{t' \\in d} f_{t',d}}"}),e.jsxs("p",{className:"text-center text-sm text-gray-500 dark:text-gray-400",children:["where ",e.jsx(r.InlineMath,{math:"f_{t,d}"})," is the raw count of term ",e.jsx(r.InlineMath,{math:"t"})," in document ",e.jsx(r.InlineMath,{math:"d"}),"."]})]}),e.jsx(c,{title:"Inverse Document Frequency (IDF)",definition:"IDF measures how informative a term is across the entire corpus. Rare terms get higher IDF scores, while common terms (like 'the') get lower scores.",id:"def-idf"}),e.jsxs("div",{className:"my-4",children:[e.jsx(r.BlockMath,{math:"\\text{idf}(t, D) = \\log \\frac{|D|}{|\\{d \\in D : t \\in d\\}|}"}),e.jsxs("p",{className:"text-center text-sm text-gray-500 dark:text-gray-400",children:["where ",e.jsx(r.InlineMath,{math:"|D|"})," is the total number of documents and the denominator counts documents containing term ",e.jsx(r.InlineMath,{math:"t"}),"."]})]}),e.jsx(j,{title:"TF-IDF Score",statement:"The TF-IDF weight of a term $t$ in document $d$ within corpus $D$ is the product: $\\text{tfidf}(t, d, D) = \\text{tf}(t, d) \\times \\text{idf}(t, D)$. This balances local importance (term frequency) with global rarity (inverse document frequency).",id:"theorem-tfidf"}),e.jsx(p,{title:"Computing TF-IDF by Hand",problem:"Given 3 documents: D1='the cat sat', D2='the dog sat', D3='the cat played'. Compute TF-IDF for 'cat' in D1.",steps:[{formula:"$\\text{tf}(\\text{cat}, D_1) = 1/3 \\approx 0.333$",explanation:'"cat" appears once out of 3 total words in D1.'},{formula:"$\\text{idf}(\\text{cat}, D) = \\log(3/2) \\approx 0.405$",explanation:'"cat" appears in 2 out of 3 documents.'},{formula:"$\\text{tfidf} = 0.333 \\times 0.405 \\approx 0.135$",explanation:"Multiply TF by IDF to get the final weight."}],id:"example-tfidf"}),e.jsx(d,{title:"bow_tfidf_sklearn.py",code:`from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "The cat chased the dog",
]

# Bag-of-Words
bow = CountVectorizer()
X_bow = bow.fit_transform(documents)
print("Vocabulary:", bow.get_feature_names_out())
print("BoW matrix (dense):")
print(X_bow.toarray())
# Each row = document, each column = word count

# TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(documents)
print("\\nTF-IDF matrix:")
print(X_tfidf.toarray().round(3))

# Document similarity using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(X_tfidf)
print("\\nCosine similarity between documents:")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        print(f"  D{i+1} vs D{j+1}: {sim_matrix[i][j]:.3f}")`,id:"code-tfidf"}),e.jsx(l,{type:"intuition",title:"Why TF-IDF Works for Search",content:"When you search for 'transformer architecture', TF-IDF naturally boosts documents that mention these specific terms frequently (high TF) while downweighting documents that merely contain common words like 'the' or 'is' (low IDF). This is exactly what makes it the backbone of traditional search engines.",id:"note-tfidf-intuition"}),e.jsx(l,{type:"note",title:"Limitations of Bag-of-Words",content:"BoW and TF-IDF ignore word order entirely: 'dog bites man' and 'man bites dog' produce identical vectors. They also cannot capture synonymy (different words, same meaning) or polysemy (same word, different meanings). Neural embeddings like Word2Vec and BERT address these shortcomings.",id:"note-bow-limitations"})]})}const ne=Object.freeze(Object.defineProperty({__proto__:null,default:F},Symbol.toStringTag,{value:"Module"}));function O(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"N-grams"}),e.jsxs("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:["N-grams are contiguous sequences of ",e.jsx(r.InlineMath,{math:"n"})," items from a text. They capture local word order and co-occurrence patterns, forming the basis of classical language models and remaining useful as features in modern NLP systems."]}),e.jsx(c,{title:"N-gram",definition:"An n-gram is a contiguous sequence of $n$ tokens from a given text. A unigram ($n=1$) is a single token, a bigram ($n=2$) is a pair of consecutive tokens, and a trigram ($n=3$) is a triple.",notation:"For a sentence $w_1, w_2, \\ldots, w_m$, the set of n-grams is $\\{(w_i, w_{i+1}, \\ldots, w_{i+n-1}) : 1 \\leq i \\leq m - n + 1\\}$.",id:"def-ngram"}),e.jsx(p,{title:"Extracting N-grams",problem:"Extract all bigrams and trigrams from the sentence: 'the cat sat on the mat'.",steps:[{formula:"Tokens: [the, cat, sat, on, the, mat]",explanation:"First tokenize the sentence into words."},{formula:"Bigrams: [(the, cat), (cat, sat), (sat, on), (on, the), (the, mat)]",explanation:"Slide a window of size 2 across the tokens."},{formula:"Trigrams: [(the, cat, sat), (cat, sat, on), (sat, on, the), (on, the, mat)]",explanation:"Slide a window of size 3 across the tokens."}],id:"example-ngrams"}),e.jsx(d,{title:"ngram_extraction.py",code:`from collections import Counter
from nltk.util import ngrams
import nltk

text = "the cat sat on the mat the cat ate the food"
tokens = text.split()

# Extract n-grams using NLTK
unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print("Bigrams:", bigrams[:5])
print("Trigrams:", trigrams[:3])

# Count n-gram frequencies
bigram_freq = Counter(bigrams)
print("\\nMost common bigrams:")
for gram, count in bigram_freq.most_common(5):
    print(f"  {gram}: {count}")

# Build a simple bigram probability table
# P(w2 | w1) = count(w1, w2) / count(w1)
unigram_freq = Counter(tokens)
print("\\nBigram probabilities P(w2 | w1):")
for (w1, w2), count in bigram_freq.most_common(5):
    prob = count / unigram_freq[w1]
    print(f"  P({w2} | {w1}) = {count}/{unigram_freq[w1]} = {prob:.3f}")`,id:"code-ngrams"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"N-gram Language Models"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["An n-gram language model estimates the probability of a word given the ",e.jsx(r.InlineMath,{math:"n-1"})," preceding words. The probability of an entire sentence is decomposed using the chain rule with the Markov assumption:"]}),e.jsx("div",{className:"my-4",children:e.jsx(r.BlockMath,{math:"P(w_1, \\ldots, w_m) \\approx \\prod_{i=1}^{m} P(w_i \\mid w_{i-n+1}, \\ldots, w_{i-1})"})}),e.jsx(d,{title:"ngram_language_model.py",code:`from collections import Counter, defaultdict
import random

# Training corpus
corpus = [
    "the cat sat on the mat",
    "the cat ate the fish",
    "the dog sat on the rug",
    "the dog chased the cat",
]

# Build bigram model with Laplace smoothing
bigram_counts = Counter()
unigram_counts = Counter()
vocab = set()

for sentence in corpus:
    tokens = ['<s>'] + sentence.split() + ['</s>']
    vocab.update(tokens)
    for i in range(len(tokens) - 1):
        bigram_counts[(tokens[i], tokens[i+1])] += 1
        unigram_counts[tokens[i]] += 1

V = len(vocab)

def bigram_prob(w2, w1, alpha=1.0):
    """Compute P(w2|w1) with Laplace smoothing."""
    return (bigram_counts[(w1, w2)] + alpha) / (unigram_counts[w1] + alpha * V)

# Generate text using the bigram model
def generate(max_len=10):
    tokens = ['<s>']
    for _ in range(max_len):
        prev = tokens[-1]
        candidates = list(vocab)
        probs = [bigram_prob(w, prev) for w in candidates]
        total = sum(probs)
        probs = [p / total for p in probs]
        next_word = random.choices(candidates, weights=probs, k=1)[0]
        if next_word == '</s>':
            break
        tokens.append(next_word)
    return ' '.join(tokens[1:])

print("Generated sentences:")
for _ in range(5):
    print(f"  {generate()}")`,id:"code-ngram-lm"}),e.jsx(u,{title:"The Curse of Dimensionality",content:"As n grows, the number of possible n-grams explodes exponentially. For a vocabulary of size V, there are V^n possible n-grams. Most will never appear in the training data, making probability estimation unreliable. This is why smoothing techniques (Laplace, Kneser-Ney) are essential.",id:"warning-sparsity"}),e.jsx(l,{type:"note",title:"Character N-grams",content:"N-grams need not be words. Character n-grams are sequences of n characters and are useful for language identification, spelling correction, and handling morphologically rich languages. The fastText model uses character n-grams to build word representations.",id:"note-char-ngrams"}),e.jsx(l,{type:"tip",title:"N-grams in Modern NLP",content:"While neural models have largely replaced n-gram language models, n-gram features remain useful as baseline features in text classification, as part of BM25 scoring in search, and in fast language identification tools like Google's CLD3.",id:"note-modern-ngrams"})]})}const ae=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function A(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Part-of-Speech Tagging"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Part-of-speech (POS) tagging assigns a grammatical category (noun, verb, adjective, etc.) to each word in a sentence. It is one of the fundamental tasks in NLP, serving as a building block for parsing, information extraction, and text understanding."}),e.jsx(c,{title:"Part-of-Speech Tag",definition:"A POS tag is a label assigned to a word indicating its syntactic role. Common tagsets include the Penn Treebank tagset (45 tags) and the Universal Dependencies tagset (17 tags). For example, 'NN' = singular noun, 'VB' = base verb, 'JJ' = adjective.",id:"def-pos"}),e.jsx(p,{title:"POS Tagging Example",problem:"Tag each word in: 'The quick brown fox jumps over the lazy dog'",steps:[{formula:"The -> DT (Determiner)",explanation:'Articles like "the", "a", "an" are determiners.'},{formula:"quick -> JJ (Adjective)",explanation:"Describes a property of the noun."},{formula:"brown -> JJ (Adjective)",explanation:'Another adjective modifying "fox".'},{formula:"fox -> NN (Noun, singular)",explanation:"The subject of the sentence."},{formula:"jumps -> VBZ (Verb, 3rd person singular present)",explanation:'The main verb, conjugated for "fox".'},{formula:"over -> IN (Preposition)",explanation:"Introduces a prepositional phrase."}],id:"example-pos"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Hidden Markov Models for POS Tagging"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Classical POS taggers use Hidden Markov Models (HMMs). The hidden states are the POS tags, and the observations are the words. The model estimates two probability distributions:"}),e.jsxs("div",{className:"my-4 space-y-2",children:[e.jsx(r.BlockMath,{math:"P(\\text{tag}_i \\mid \\text{tag}_{i-1}) \\quad \\text{(transition probability)}"}),e.jsx(r.BlockMath,{math:"P(\\text{word}_i \\mid \\text{tag}_i) \\quad \\text{(emission probability)}"})]}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The Viterbi algorithm finds the most likely tag sequence by dynamic programming:"}),e.jsx("div",{className:"my-4",children:e.jsx(r.BlockMath,{math:"\\hat{t}_1^n = \\arg\\max_{t_1^n} \\prod_{i=1}^{n} P(w_i \\mid t_i) \\cdot P(t_i \\mid t_{i-1})"})}),e.jsx(d,{title:"pos_tagging_spacy.py",code:`import spacy

# Load the English model (small)
nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog"
doc = nlp(text)

# Display POS tags
print(f"{'Token':<12} {'POS':<8} {'Fine POS':<8} {'Description'}")
print("-" * 55)
for token in doc:
    print(f"{token.text:<12} {token.pos_:<8} {token.tag_:<8} {spacy.explain(token.tag_)}")

# POS tag distribution in a longer text
long_text = """
Natural language processing enables computers to understand human language.
Machine learning models learn patterns from large datasets of text.
Transformers revolutionized the field with attention mechanisms.
"""
doc2 = nlp(long_text)
from collections import Counter
pos_counts = Counter(token.pos_ for token in doc2 if not token.is_space)
print("\\nPOS distribution:")
for pos, count in pos_counts.most_common():
    print(f"  {pos:<8} {count}")`,id:"code-pos-spacy"}),e.jsx(l,{type:"intuition",title:"Why POS Tagging Is Hard",content:"Many words are ambiguous: 'run' can be a verb ('I run daily') or a noun ('a morning run'). 'Back' can be a noun, verb, adjective, or adverb. Context is essential. The word 'flies' in 'time flies like an arrow' vs 'fruit flies like a banana' has completely different tags.",id:"note-ambiguity"}),e.jsx(d,{title:"pos_ambiguity.py",code:`import spacy
nlp = spacy.load("en_core_web_sm")

# The same word gets different POS tags in different contexts
examples = [
    "I need to book a flight",        # book = VERB
    "I read an interesting book",      # book = NOUN
    "They will fly to Paris",          # fly = VERB
    "A fly landed on the table",       # fly = NOUN
    "The old man the boats",           # old = NOUN (garden path!)
]

for sent in examples:
    doc = nlp(sent)
    tags = [(t.text, t.pos_) for t in doc]
    print(f"{sent}")
    print(f"  Tags: {tags}\\n")`,id:"code-pos-ambiguity"}),e.jsx(u,{title:"Tagset Differences",content:"Different POS tagsets exist: Penn Treebank (PTB) uses fine-grained tags like VBZ, VBD, VBG for verb forms, while Universal Dependencies (UD) uses coarser tags like VERB. Always check which tagset your tools use, as downstream tasks may expect a specific one.",id:"warning-tagsets"}),e.jsx(l,{type:"historical",title:"From Rules to Neural Models",content:"Early POS taggers (1960s) used hand-written rules. HMM taggers (1980s-90s) achieved ~96% accuracy. The Brill tagger (1992) used transformation-based learning. Modern neural taggers using BiLSTMs and Transformers achieve ~97.5% accuracy, approaching the human inter-annotator agreement ceiling of ~97%.",id:"note-pos-history"})]})}const ie=Object.freeze(Object.defineProperty({__proto__:null,default:A},Symbol.toStringTag,{value:"Module"}));function D(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Named Entity Recognition"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Named Entity Recognition (NER) identifies and classifies named entities in text into predefined categories such as persons, organizations, locations, dates, and more. It is a core information extraction task used in search, question answering, and knowledge graph construction."}),e.jsx(c,{title:"Named Entity Recognition (NER)",definition:"NER is a sequence labeling task that identifies spans of text referring to real-world entities and classifies them into categories. Standard entity types include PERSON, ORGANIZATION (ORG), LOCATION (LOC), DATE, TIME, MONEY, and PERCENT.",id:"def-ner"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"BIO Tagging Scheme"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"NER is typically framed as a token-level classification task using the BIO (Beginning, Inside, Outside) tagging scheme. Each token receives a tag indicating whether it begins an entity (B-TYPE), continues one (I-TYPE), or is outside any entity (O)."}),e.jsx(p,{title:"BIO Tagging",problem:"Apply BIO tags to: 'Barack Obama visited New York City on Friday'",steps:[{formula:"Barack -> B-PER",explanation:"Begins a PERSON entity."},{formula:"Obama -> I-PER",explanation:"Continues the PERSON entity."},{formula:"visited -> O",explanation:"Not part of any entity."},{formula:"New -> B-LOC",explanation:"Begins a LOCATION entity."},{formula:"York -> I-LOC",explanation:"Continues the LOCATION entity."},{formula:"City -> I-LOC",explanation:"Still part of the LOCATION entity."}],id:"example-bio"}),e.jsx(d,{title:"ner_spacy.py",code:`import spacy

nlp = spacy.load("en_core_web_sm")

text = """Apple Inc. was founded by Steve Jobs in Cupertino, California
in 1976. The company is now worth over $2.8 trillion and employs
more than 160,000 people worldwide."""

doc = nlp(text)

# Extract named entities
print(f"{'Entity':<25} {'Label':<12} {'Description'}")
print("-" * 60)
for ent in doc.ents:
    print(f"{ent.text:<25} {ent.label_:<12} {spacy.explain(ent.label_)}")

# Visualize entity spans with character offsets
print("\\nEntity spans:")
for ent in doc.ents:
    print(f"  [{ent.start_char}:{ent.end_char}] '{ent.text}' ({ent.label_})")

# Count entity types
from collections import Counter
type_counts = Counter(ent.label_ for ent in doc.ents)
print("\\nEntity type distribution:")
for label, count in type_counts.most_common():
    print(f"  {label}: {count}")`,id:"code-ner-spacy"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"NER as Sequence Labeling"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["Formally, NER finds the tag sequence ",e.jsx(r.InlineMath,{math:"\\hat{y}_1^n"})," that maximizes:"]}),e.jsx("div",{className:"my-4",children:e.jsx(r.BlockMath,{math:"\\hat{y}_1^n = \\arg\\max_{y_1^n} P(y_1, \\ldots, y_n \\mid x_1, \\ldots, x_n)"})}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Classical approaches use Conditional Random Fields (CRFs) to model the joint probability, capturing dependencies between adjacent tags (e.g., I-PER should not follow B-LOC)."}),e.jsx(d,{title:"ner_custom_rules.py",code:`import re

# Simple rule-based NER using regex patterns
patterns = {
    'EMAIL': r'\b[w.+-]+@[w-]+.[w.-]+\b',
    'PHONE': r'\bd{3}[-.]?d{3}[-.]?d{4}\b',
    'URL': r'https?://[w/-?=%.]+.[w/-?=%.]+',
    'DATE': r'\bd{1,2}/d{1,2}/d{2,4}\b',
    'MONEY': r'$[d,]+.?d*',
}

text = """Contact john.doe@example.com or call 555-123-4567.
Visit https://openai.com for details. Meeting on 3/15/2024.
Budget: $1,250,000 allocated for Q2."""

print("Rule-based entity extraction:")
for entity_type, pattern in patterns.items():
    matches = re.findall(pattern, text)
    if matches:
        print(f"  {entity_type}: {matches}")

# Compare: spaCy finds semantic entities, regex finds patterns
# Both approaches are useful and often combined in practice`,id:"code-ner-rules"}),e.jsx(l,{type:"tip",title:"Evaluation Metrics for NER",content:"NER is evaluated using entity-level precision, recall, and F1 score. An entity is correct only if both the span boundaries AND the type label match exactly. Token-level accuracy can be misleadingly high because most tokens are 'O' (outside any entity).",id:"note-ner-eval"}),e.jsx(u,{title:"Nested and Overlapping Entities",content:"Standard BIO tagging cannot handle nested entities. In 'New York University', 'New York' is a LOC inside the ORG 'New York University'. Specialized approaches like span-based models or multi-layer tagging are needed for nested NER.",id:"warning-nested"}),e.jsx(l,{type:"historical",title:"NER Timeline",content:"NER emerged from the MUC conferences (1990s). Early systems were rule-based. CRF-based taggers (Lafferty et al., 2001) dominated for a decade. BiLSTM-CRF models (Lample et al., 2016) set new benchmarks. Modern Transformer-based NER (BERT fine-tuning) achieves F1 scores above 92% on CoNLL-2003.",id:"note-ner-history"})]})}const se=Object.freeze(Object.defineProperty({__proto__:null,default:D},Symbol.toStringTag,{value:"Module"}));function I(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Dependency Parsing"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Dependency parsing reveals the grammatical structure of a sentence by establishing directed relationships between words. Each word depends on exactly one other word (its head), forming a tree structure that captures who does what to whom."}),e.jsx(c,{title:"Dependency Tree",definition:"A dependency tree is a directed tree where each node is a word in the sentence, and each edge represents a grammatical relation (dependency) from a head word to a dependent word. The root of the tree is typically the main verb.",notation:"A dependency relation is written as $\\text{head} \\xrightarrow{\\text{rel}} \\text{dependent}$. For example, $\\text{chased} \\xrightarrow{\\text{nsubj}} \\text{cat}$ means 'cat' is the nominal subject of 'chased'.",id:"def-dep-tree"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Universal Dependencies"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The Universal Dependencies (UD) framework defines a cross-linguistically consistent set of dependency relations. Key relations include:"}),e.jsxs("ul",{className:"ml-6 list-disc space-y-1 text-gray-700 dark:text-gray-300",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"nsubj"})," - nominal subject"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"obj"})," - direct object"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"amod"})," - adjectival modifier"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"det"})," - determiner"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"advmod"})," - adverbial modifier"]}),e.jsxs("li",{children:[e.jsx("strong",{children:"prep / obl"})," - prepositional/oblique modifier"]})]}),e.jsx(p,{title:"Dependency Parse Example",problem:"Parse the dependencies in: 'The large cat quickly chased the small mouse'",steps:[{formula:"chased is the ROOT",explanation:"The main verb is the root of the tree."},{formula:"chased -nsubj-> cat",explanation:'"cat" is the nominal subject of "chased".'},{formula:"cat -det-> The",explanation:'"The" is the determiner of "cat".'},{formula:"cat -amod-> large",explanation:'"large" is an adjectival modifier of "cat".'},{formula:"chased -obj-> mouse",explanation:'"mouse" is the direct object of "chased".'},{formula:"chased -advmod-> quickly",explanation:'"quickly" is an adverbial modifier of "chased".'}],id:"example-dep-parse"}),e.jsx(d,{title:"dependency_parsing_spacy.py",code:`import spacy

nlp = spacy.load("en_core_web_sm")

sentence = "The quick brown fox jumps over the lazy dog"
doc = nlp(sentence)

# Display dependency tree
print(f"{'Token':<10} {'Dep':<10} {'Head':<10} {'Children'}")
print("-" * 55)
for token in doc:
    children = [child.text for child in token.children]
    print(f"{token.text:<10} {token.dep_:<10} {token.head.text:<10} {children}")

# Find the root and traverse the tree
root = [token for token in doc if token.dep_ == "ROOT"][0]
print(f"\\nRoot: '{root.text}' ({root.pos_})")

# Extract subject-verb-object triples
def extract_svo(doc):
    """Extract subject-verb-object triples from a parsed sentence."""
    triples = []
    for token in doc:
        if token.dep_ == "ROOT":
            verb = token
            subj = None
            obj = None
            for child in verb.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = child
                elif child.dep_ in ("dobj", "obj"):
                    obj = child
            if subj and obj:
                triples.append((subj.text, verb.text, obj.text))
    return triples

sentences = [
    "The cat chased the mouse",
    "Scientists discovered a new species",
    "The student wrote an excellent paper",
]

for sent in sentences:
    doc = nlp(sent)
    triples = extract_svo(doc)
    print(f"'{sent}' -> SVO: {triples}")`,id:"code-dep-parsing"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Parsing Algorithms"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Two main families of algorithms exist for dependency parsing:"}),e.jsxs("ul",{className:"ml-6 list-disc space-y-2 text-gray-700 dark:text-gray-300",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Transition-based parsing"})," uses a stack and buffer with shift/reduce actions. It runs in ",e.jsx(r.InlineMath,{math:"O(n)"})," time but makes greedy local decisions. The arc-standard and arc-eager systems are popular variants."]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Graph-based parsing"})," scores all possible edges and finds the maximum spanning tree. Eisner's algorithm runs in ",e.jsx(r.InlineMath,{math:"O(n^3)"})," for projective trees. This approach considers the global structure but is slower."]})]}),e.jsx(l,{type:"intuition",title:"Why Dependencies Matter for LLMs",content:"While LLMs do not explicitly build dependency trees, they implicitly learn syntactic structure through attention patterns. Research has shown that specific attention heads in BERT and GPT models correspond to dependency relations, suggesting that understanding syntax is a natural byproduct of language modeling.",id:"note-dep-llm"}),e.jsx(u,{title:"Non-Projective Dependencies",content:"In some languages (Czech, Dutch, German), dependency arcs can cross each other (non-projective trees). Standard shift-reduce parsers cannot produce non-projective trees. Special algorithms like the Chu-Liu-Edmonds algorithm or swap-based transitions are needed.",id:"warning-nonprojective"}),e.jsx(l,{type:"historical",title:"Parsing History",content:"Dependency grammar dates to Lucien Tesniere (1959). Computational dependency parsing took off with Nivre's arc-eager parser (2003) and McDonald's MST parser (2005). Chen and Manning's neural dependency parser (2014) showed that neural networks could replace hand-crafted features, paving the way for modern parsers.",id:"note-parsing-history"})]})}const re=Object.freeze(Object.defineProperty({__proto__:null,default:I},Symbol.toStringTag,{value:"Module"}));function R(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Sentiment Analysis Basics"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Sentiment analysis determines the emotional tone or opinion expressed in text. It ranges from simple positive/negative classification to fine-grained analysis of aspect-level sentiment and emotion detection. It is one of the most commercially important NLP tasks."}),e.jsx(c,{title:"Sentiment Analysis",definition:"Sentiment analysis (or opinion mining) is the task of identifying and extracting subjective information from text. At its simplest, it classifies text as positive, negative, or neutral. More advanced forms detect the target of sentiment, the holder of the opinion, and the intensity of emotion.",id:"def-sentiment"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Lexicon-Based Methods"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The simplest approach uses a sentiment lexicon: a dictionary that maps words to sentiment scores. The overall sentiment of a text is computed by aggregating individual word scores."}),e.jsx(d,{title:"lexicon_sentiment.py",code:`# Simple lexicon-based sentiment analysis
sentiment_lexicon = {
    "good": 1.0, "great": 1.5, "excellent": 2.0, "amazing": 2.0,
    "love": 1.5, "wonderful": 1.5, "fantastic": 2.0, "best": 1.5,
    "bad": -1.0, "terrible": -2.0, "awful": -2.0, "worst": -2.0,
    "hate": -1.5, "horrible": -2.0, "poor": -1.0, "boring": -1.0,
    "not": None,  # negation modifier
}

def lexicon_sentiment(text):
    """Simple lexicon sentiment with basic negation handling."""
    words = text.lower().split()
    score = 0.0
    negate = False
    for word in words:
        if word == "not" or word == "n't":
            negate = True
            continue
        if word in sentiment_lexicon and sentiment_lexicon[word] is not None:
            word_score = sentiment_lexicon[word]
            if negate:
                word_score *= -0.5  # Flip and dampen
                negate = False
            score += word_score
        else:
            negate = False  # Reset negation after non-sentiment word
    return score

reviews = [
    "This movie was great and the acting was excellent",
    "Terrible film with awful dialogue and bad pacing",
    "The food was not bad but not great either",
    "I love this amazing wonderful product",
]

for review in reviews:
    score = lexicon_sentiment(review)
    label = "POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else "NEUTRAL"
    print(f"[{score:+.1f}] {label}: '{review}'")`,id:"code-lexicon"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Machine Learning Approaches"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"ML-based sentiment analysis treats the problem as text classification. A document is represented as a feature vector (e.g., TF-IDF), and a classifier (Naive Bayes, SVM, or logistic regression) predicts the sentiment label."}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"For Naive Bayes, the predicted class is:"}),e.jsx("div",{className:"my-4",children:e.jsx(r.BlockMath,{math:"\\hat{c} = \\arg\\max_{c} P(c) \\prod_{i=1}^{n} P(w_i \\mid c)"})}),e.jsx(d,{title:"ml_sentiment.py",code:`from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Sample training data
texts = [
    "I love this movie, it was fantastic",
    "Great film with wonderful performances",
    "The best movie I have seen this year",
    "Excellent acting and a compelling story",
    "This movie was terrible and boring",
    "Awful waste of time, worst film ever",
    "Bad acting, poor script, horrible movie",
    "I hated every minute of this film",
]
labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1=positive, 0=negative

# TF-IDF features
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X = vectorizer.fit_transform(texts)

# Train classifiers
nb = MultinomialNB()
nb.fit(X, labels)

lr = LogisticRegression()
lr.fit(X, labels)

# Predict on new reviews
new_reviews = [
    "A wonderful and great experience",
    "Terrible and boring, do not watch",
    "The movie was okay, nothing special",
]

X_new = vectorizer.transform(new_reviews)
for review, nb_pred, lr_pred in zip(new_reviews, nb.predict(X_new), lr.predict(X_new)):
    print(f"NB={['NEG','POS'][nb_pred]}, LR={['NEG','POS'][lr_pred]}: '{review}'")`,id:"code-ml-sentiment"}),e.jsx(p,{title:"VADER: Rule-Based Sentiment for Social Media",problem:"VADER (Valence Aware Dictionary and sEntiment Reasoner) handles emoticons, slang, and capitalization. How does it score: 'This movie is GREAT!!! :)' ?",steps:[{formula:'"GREAT" -> boosted score (all caps)',explanation:"VADER boosts sentiment for capitalized words."},{formula:'"!!!" -> intensified',explanation:"Exclamation marks increase the intensity."},{formula:'":)" -> positive emoticon',explanation:"Emoticons have predefined sentiment values."},{formula:"Compound score: ~0.82 (strongly positive)",explanation:"VADER combines and normalizes all signals into a compound score in [-1, 1]."}],id:"example-vader"}),e.jsx(u,{title:"Sarcasm and Context",content:"Lexicon and simple ML methods fail on sarcasm ('Oh great, another meeting'), domain-specific language ('This drug killed the infection' is positive in medicine), and implicit sentiment ('The battery lasted 20 minutes' is negative without using negative words). These require deeper contextual understanding.",id:"warning-sarcasm"}),e.jsx(l,{type:"note",title:"From Classical to Neural Sentiment",content:"Modern sentiment analysis uses fine-tuned Transformer models (BERT, RoBERTa) that achieve state-of-the-art accuracy by leveraging contextual embeddings. LLMs can also perform sentiment analysis via zero-shot prompting, often matching or exceeding supervised baselines without any training data.",id:"note-neural-sentiment"})]})}const oe=Object.freeze(Object.defineProperty({__proto__:null,default:R},Symbol.toStringTag,{value:"Module"}));function U(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"What is a Language Model?"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"A language model assigns a probability to a sequence of words. This simple idea is the mathematical foundation of all modern LLMs: GPT, LLaMA, Claude, and every other generative text model. Understanding what language models compute is essential to understanding how they work."}),e.jsx(c,{title:"Language Model",definition:"A language model is a probability distribution over sequences of tokens. Given a sequence $w_1, w_2, \\ldots, w_n$, a language model assigns a probability $P(w_1, w_2, \\ldots, w_n)$ that measures how likely this sequence is under the model's learned distribution.",notation:"$P(w_1, w_2, \\ldots, w_n)$ or equivalently $P(\\mathbf{w})$ where $\\mathbf{w}$ is a token sequence.",id:"def-lm"}),e.jsx(j,{title:"Chain Rule of Probability",statement:"Any joint probability over a sequence can be decomposed exactly using the chain rule: $P(w_1, w_2, \\ldots, w_n) = \\prod_{i=1}^{n} P(w_i \\mid w_1, w_2, \\ldots, w_{i-1})$. This is not an approximation; it is an identity from probability theory.",id:"theorem-chain-rule"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The chain rule tells us that a language model can be equivalently viewed as a next-token predictor. At each step, the model predicts the distribution over the next token given all preceding tokens:"}),e.jsx("div",{className:"my-4",children:e.jsx(r.BlockMath,{math:"P(w_t \\mid w_1, \\ldots, w_{t-1})"})}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"This is precisely what GPT and other autoregressive models learn to do."}),e.jsx(p,{title:"Computing Sequence Probability",problem:"A language model gives these next-token probabilities. What is $P(\\text{'the cat sat'})$?",steps:[{formula:"$P(\\text{the}) = 0.05$",explanation:'Probability of "the" as the first word.'},{formula:"$P(\\text{cat} \\mid \\text{the}) = 0.02$",explanation:'Probability of "cat" following "the".'},{formula:"$P(\\text{sat} \\mid \\text{the cat}) = 0.08$",explanation:'Probability of "sat" given "the cat".'},{formula:"$P(\\text{the cat sat}) = 0.05 \\times 0.02 \\times 0.08 = 0.00008$",explanation:"Multiply all conditional probabilities by the chain rule."}],id:"example-seq-prob"}),e.jsx(d,{title:"language_model_basics.py",code:`import numpy as np

# A simple demonstration of language modeling
# A language model defines P(next_token | context)

# Vocabulary
vocab = ["the", "cat", "dog", "sat", "ran", "on", "mat", "<end>"]
V = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}

# A toy conditional probability table: P(next | prev)
# Each row sums to 1.0
cond_probs = {
    "<start>": {"the": 0.6, "cat": 0.1, "dog": 0.1, "sat": 0.05,
                "ran": 0.05, "on": 0.02, "mat": 0.02, "<end>": 0.06},
    "the":     {"cat": 0.3, "dog": 0.3, "mat": 0.2, "the": 0.01,
                "sat": 0.05, "ran": 0.05, "on": 0.05, "<end>": 0.04},
    "cat":     {"sat": 0.4, "ran": 0.3, "on": 0.1, "the": 0.05,
                "cat": 0.01, "dog": 0.01, "mat": 0.03, "<end>": 0.1},
    "sat":     {"on": 0.6, "the": 0.1, "<end>": 0.2, "cat": 0.02,
                "dog": 0.02, "sat": 0.01, "ran": 0.02, "mat": 0.03},
}

def sequence_probability(tokens):
    """Compute P(w1, w2, ..., wn) using the chain rule."""
    prob = 1.0
    prev = "<start>"
    for token in tokens:
        if prev in cond_probs and token in cond_probs[prev]:
            p = cond_probs[prev][token]
        else:
            p = 1e-6  # Smoothing for unseen transitions
        prob *= p
        prev = token
    return prob

# Compare probabilities of different sequences
sequences = [
    ["the", "cat", "sat", "on"],
    ["the", "dog", "ran"],
    ["cat", "the", "mat", "sat"],  # Ungrammatical
    ["sat", "cat", "on", "the"],   # Ungrammatical
]

print("Sequence probabilities:")
for seq in sequences:
    p = sequence_probability(seq)
    print(f"  P({' '.join(seq)}) = {p:.8f}")

# The model assigns higher probability to grammatical sequences!`,id:"code-lm-basics"}),e.jsx(l,{type:"intuition",title:"Language Models as World Models",content:"A sufficiently good language model must implicitly capture facts, reasoning, and world knowledge. To correctly predict that 'The capital of France is ___' should be completed with 'Paris', the model must 'know' geography. This is why scaling language models has led to emergent capabilities far beyond simple text prediction.",id:"note-lm-world-model"}),e.jsx(l,{type:"note",title:"Three Uses of Language Models",content:"Language models serve three main purposes: (1) Scoring - evaluating how likely or fluent a sentence is (used in speech recognition, machine translation); (2) Generation - sampling new text by repeatedly predicting the next token; (3) Representation - using internal states as features for downstream tasks (like BERT embeddings).",id:"note-lm-uses"}),e.jsx(l,{type:"historical",title:"The Language Modeling Hypothesis",content:"Shannon (1948) first framed language as a stochastic process and proposed measuring its entropy. Jelinek at IBM (1970s-80s) built n-gram language models for speech recognition, famously stating 'Every time I fire a linguist, the performance of the speech recognizer goes up.' The modern LLM era began with GPT (2018), showing that large-scale language modeling alone can produce powerful general-purpose AI systems.",id:"note-lm-history"})]})}const le=Object.freeze(Object.defineProperty({__proto__:null,default:U},Symbol.toStringTag,{value:"Module"}));function W(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Markov Models"}),e.jsxs("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:["The chain rule decomposes sequence probabilities exactly, but estimating the full conditional ",e.jsx(r.InlineMath,{math:"P(w_t \\mid w_1, \\ldots, w_{t-1})"})," is intractable for long sequences. Markov models make this feasible by assuming that only recent context matters."]}),e.jsx(c,{title:"Markov Assumption",definition:"The $k$-th order Markov assumption states that the probability of a token depends only on the preceding $k$ tokens: $P(w_t \\mid w_1, \\ldots, w_{t-1}) \\approx P(w_t \\mid w_{t-k}, \\ldots, w_{t-1})$. A bigram model uses $k=1$, a trigram model uses $k=2$.",id:"def-markov"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Bigram Model"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The simplest Markov language model conditions only on the immediately preceding word:"}),e.jsx("div",{className:"my-4",children:e.jsx(r.BlockMath,{math:"P(w_1, \\ldots, w_n) \\approx \\prod_{i=1}^{n} P(w_i \\mid w_{i-1})"})}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Bigram probabilities are estimated from counts using maximum likelihood estimation (MLE):"}),e.jsx("div",{className:"my-4",children:e.jsx(r.BlockMath,{math:"P_{\\text{MLE}}(w_i \\mid w_{i-1}) = \\frac{C(w_{i-1}, w_i)}{C(w_{i-1})}"})}),e.jsx(p,{title:"Bigram MLE",problem:"Given the corpus: 'the cat sat . the cat ate . the dog sat .', compute $P(\\\\text{sat} \\\\mid \\\\text{cat})$.",steps:[{formula:"$C(\\text{cat}, \\text{sat}) = 1$",explanation:'The bigram "cat sat" appears once.'},{formula:"$C(\\text{cat}) = 2$",explanation:'"cat" appears twice as a context word (before "sat" and "ate").'},{formula:"$P(\\text{sat} \\mid \\text{cat}) = 1/2 = 0.5$",explanation:"Divide bigram count by unigram count of the context word."}],id:"example-bigram-mle"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Smoothing"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"MLE assigns zero probability to unseen n-grams, which is catastrophic: a single unseen bigram makes the entire sequence probability zero. Smoothing techniques redistribute probability mass to unseen events."}),e.jsx(j,{title:"Laplace (Add-1) Smoothing",statement:"Add-1 smoothing adds 1 to every n-gram count: $P_{\\text{Laplace}}(w_i \\mid w_{i-1}) = \\frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}$ where $V$ is the vocabulary size. This ensures no probability is ever zero.",id:"theorem-laplace"}),e.jsx(d,{title:"markov_language_model.py",code:`from collections import Counter, defaultdict
import random

# Training corpus
corpus = """the cat sat on the mat . the cat ate the fish .
the dog sat on the rug . the dog chased the cat .
the bird flew over the house . the cat watched the bird ."""

# Tokenize
sentences = [s.strip().split() for s in corpus.split('.') if s.strip()]
# Add start/end tokens
sentences = [['<s>'] + s + ['</s>'] for s in sentences]

# Count unigrams and bigrams
unigram_counts = Counter()
bigram_counts = Counter()
for sent in sentences:
    for i in range(len(sent)):
        unigram_counts[sent[i]] += 1
        if i > 0:
            bigram_counts[(sent[i-1], sent[i])] += 1

vocab = set(unigram_counts.keys())
V = len(vocab)

def bigram_prob(w2, w1, alpha=1.0):
    """P(w2 | w1) with Laplace smoothing."""
    return (bigram_counts[(w1, w2)] + alpha) / (unigram_counts[w1] + alpha * V)

# Show some bigram probabilities
print("Bigram probabilities:")
contexts = ["<s>", "the", "cat", "dog"]
for ctx in contexts:
    top = sorted(vocab, key=lambda w: bigram_prob(w, ctx), reverse=True)[:3]
    probs = [f"{w}:{bigram_prob(w, ctx):.3f}" for w in top]
    print(f"  P(? | {ctx}): {', '.join(probs)}")

# Generate text
def generate_bigram(max_len=15):
    tokens = ['<s>']
    for _ in range(max_len):
        prev = tokens[-1]
        candidates = list(vocab)
        probs = [bigram_prob(w, prev) for w in candidates]
        next_word = random.choices(candidates, weights=probs, k=1)[0]
        if next_word == '</s>':
            break
        tokens.append(next_word)
    return ' '.join(tokens[1:])

print("\\nGenerated sentences (bigram model):")
for i in range(5):
    print(f"  {i+1}. {generate_bigram()}")`,id:"code-markov"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Trigram and Higher-Order Models"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Trigram models condition on two preceding words, capturing more context:"}),e.jsx("div",{className:"my-4",children:e.jsx(r.BlockMath,{math:"P(w_i \\mid w_{i-2}, w_{i-1}) = \\frac{C(w_{i-2}, w_{i-1}, w_i)}{C(w_{i-2}, w_{i-1})}"})}),e.jsx(u,{title:"The Sparsity Problem",content:"Higher-order n-grams are exponentially sparser. With a vocabulary of 50,000 words, there are 50,000^3 = 1.25 x 10^14 possible trigrams. Even massive corpora will only observe a tiny fraction. This is the fundamental limitation that neural language models overcome by learning continuous representations.",id:"warning-sparsity"}),e.jsx(l,{type:"tip",title:"Interpolation and Backoff",content:"Practical n-gram models use interpolation (a weighted mix of unigram, bigram, and trigram probabilities) or backoff (use the trigram if seen, otherwise fall back to bigram, then unigram). Kneser-Ney smoothing, which uses a sophisticated backoff distribution, is considered the best classical smoothing method.",id:"note-interpolation"}),e.jsx(l,{type:"intuition",title:"Markov Models vs. Transformers",content:"An n-gram model with context k can only 'see' k tokens back. A Transformer with context window C can attend to all C previous tokens. GPT-4 with a 128k context window is like a 128,000-gram model, except it generalizes through learned parameters instead of memorizing counts.",id:"note-markov-vs-transformer"})]})}const de=Object.freeze(Object.defineProperty({__proto__:null,default:W},Symbol.toStringTag,{value:"Module"}));function V(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Perplexity and Evaluation"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:'How do we measure whether one language model is better than another? Perplexity is the standard intrinsic evaluation metric. It quantifies how "surprised" a model is by test data -- lower perplexity means the model predicts the data better.'}),e.jsx(c,{title:"Cross-Entropy",definition:"The cross-entropy of a language model $P_\\theta$ on a test sequence $w_1, \\ldots, w_N$ is: $H(P_\\theta) = -\\frac{1}{N} \\sum_{i=1}^{N} \\log_2 P_\\theta(w_i \\mid w_1, \\ldots, w_{i-1})$. It measures the average number of bits needed to encode each token under the model.",notation:"$H(P_\\theta)$ is measured in bits when using $\\log_2$, or in nats when using $\\ln$.",id:"def-cross-entropy"}),e.jsx(c,{title:"Perplexity",definition:"Perplexity is the exponentiated cross-entropy: $\\text{PPL}(P_\\theta) = 2^{H(P_\\theta)}$. Equivalently, it is the inverse geometric mean probability assigned to each token.",notation:"$\\text{PPL} = 2^{-\\frac{1}{N} \\sum_{i=1}^{N} \\log_2 P_\\theta(w_i \\mid w_{<i})}$",id:"def-perplexity"}),e.jsx(j,{title:"Perplexity as Weighted Branching Factor",statement:"Perplexity can be interpreted as the effective vocabulary size the model is choosing from at each step. A perplexity of 100 means the model is, on average, as uncertain as if it were choosing uniformly among 100 equally likely tokens.",id:"theorem-ppl-interp"}),e.jsx(p,{title:"Computing Perplexity",problem:"A bigram model assigns these probabilities to a test sentence 'the cat sat': $P(\\\\text{the}) = 0.1$, $P(\\\\text{cat}|\\\\text{the}) = 0.05$, $P(\\\\text{sat}|\\\\text{cat}) = 0.2$. What is the perplexity?",steps:[{formula:"$\\log_2 P = \\log_2(0.1) + \\log_2(0.05) + \\log_2(0.2)$",explanation:"Sum the log probabilities of each token."},{formula:"$= -3.322 + (-4.322) + (-2.322) = -9.966$",explanation:"Compute each log base 2 value."},{formula:"$H = -\\frac{1}{3}(-9.966) = 3.322$ bits",explanation:"Divide by the number of tokens N=3."},{formula:"$\\text{PPL} = 2^{3.322} = 10.0$",explanation:"Exponentiate to get perplexity. The model is as uncertain as choosing from 10 options."}],id:"example-ppl"}),e.jsx(d,{title:"perplexity_computation.py",code:`import numpy as np
from collections import Counter

def compute_perplexity(log_probs):
    """
    Compute perplexity from a list of log2 probabilities.
    PPL = 2^(-1/N * sum(log2(P)))
    """
    N = len(log_probs)
    cross_entropy = -np.sum(log_probs) / N
    perplexity = 2 ** cross_entropy
    return perplexity, cross_entropy

# Example: compare two models on the same test data
# Model A: assigns higher probabilities (better model)
model_a_probs = [0.1, 0.05, 0.2, 0.15, 0.3]
# Model B: assigns lower probabilities (worse model)
model_b_probs = [0.02, 0.01, 0.05, 0.03, 0.08]

log_probs_a = np.log2(model_a_probs)
log_probs_b = np.log2(model_b_probs)

ppl_a, ce_a = compute_perplexity(log_probs_a)
ppl_b, ce_b = compute_perplexity(log_probs_b)

print(f"Model A: perplexity = {ppl_a:.1f}, cross-entropy = {ce_a:.3f} bits")
print(f"Model B: perplexity = {ppl_b:.1f}, cross-entropy = {ce_b:.3f} bits")
print(f"Model A is {ppl_b/ppl_a:.1f}x better (lower PPL = better)")

# Real-world perplexity benchmarks (approximate)
benchmarks = {
    "Trigram model (1990s)":       220,
    "LSTM (2016)":                 82,
    "GPT-2 (2019)":               35,
    "GPT-3 (2020)":               20,
    "GPT-4 class (2023)":         8,
    "Uniform random (50k vocab)":  50000,
}

print("\\nHistorical perplexity on Penn Treebank (approx):")
for model, ppl in sorted(benchmarks.items(), key=lambda x: -x[1]):
    bar = "#" * int(np.log2(ppl))
    print(f"  {model:<30s} PPL={ppl:<8} {bar}")`,id:"code-perplexity"}),e.jsx(u,{title:"Perplexity Comparisons Require Same Tokenization",content:"You can only compare perplexity between models that use the same vocabulary and tokenization. A character-level model will have lower perplexity per character but higher perplexity per word than a word-level model. Always specify the tokenization when reporting perplexity.",id:"warning-ppl-comparison"}),e.jsx(l,{type:"tip",title:"Bits-Per-Character (BPC)",content:"To compare models with different tokenizations, normalize by the number of characters instead of tokens. Bits-per-character (BPC) = total cross-entropy / number of characters. This gives a tokenization-independent measure of model quality.",id:"note-bpc"}),e.jsx(l,{type:"note",title:"Beyond Perplexity",content:"Perplexity measures how well a model predicts text, but it does not directly measure downstream task performance. A model with lower perplexity is not always better at summarization, translation, or reasoning. Modern LLM evaluation increasingly relies on benchmarks like MMLU, HumanEval, and human preference ratings rather than perplexity alone.",id:"note-beyond-ppl"})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:V},Symbol.toStringTag,{value:"Module"}));function G(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Statistical vs Neural Language Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Language modeling has undergone a dramatic transformation from count-based statistical methods to neural networks. Understanding this evolution explains why modern LLMs are so much more capable than their predecessors and what fundamental problems neural approaches solve."}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Statistical Language Models"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Statistical LMs estimate probabilities directly from corpus counts. N-gram models are the canonical example. Their key limitation is that they treat each word as an independent atomic symbol with no notion of similarity between words."}),e.jsx(c,{title:"The Sparsity Problem",definition:"In n-gram models, the number of possible n-grams grows as $V^n$ where $V$ is vocabulary size. For a 50,000-word vocabulary, there are $2.5 \\times 10^9$ possible bigrams, most of which will never appear in training data. This means most probability estimates are based on zero counts.",id:"def-sparsity"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Neural Language Models"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:'Neural LMs represent words as dense vectors (embeddings) in a continuous space. Similar words have similar vectors, enabling the model to generalize: if it learns that "the cat sat on the mat" is likely, it can infer that "the dog sat on the rug" is also likely, because "cat" and "dog" have similar embeddings.'}),e.jsx(c,{title:"Neural Language Model",definition:"A neural language model parameterizes $P(w_t \\mid w_{t-k}, \\ldots, w_{t-1})$ using a neural network with parameters $\\theta$. Words are embedded in $\\mathbb{R}^d$, and the network learns a smooth function from context embeddings to a probability distribution over the vocabulary.",notation:"$P_\\theta(w_t \\mid w_{<t}) = \\text{softmax}(f_\\theta(\\mathbf{e}_{w_{<t}}))$ where $\\mathbf{e}$ are learned embeddings.",id:"def-neural-lm"}),e.jsx(p,{title:"Generalization Through Embeddings",problem:"A statistical model trained on 'The cat sits on the mat' assigns zero probability to 'The dog sits on the rug'. How does a neural model handle this?",steps:[{formula:"Embeddings: cat ~ dog (both animals)",explanation:'The model learns that "cat" and "dog" have similar vector representations.'},{formula:"Embeddings: mat ~ rug (both floor coverings)",explanation:'Similarly, "mat" and "rug" are nearby in embedding space.'},{formula:"$P(\\text{dog sits on the rug}) > 0$",explanation:"Because the context embeddings are similar, the model assigns non-zero probability to the unseen but analogous sentence."}],id:"example-generalization"}),e.jsx(d,{title:"lm_evolution_comparison.py",code:`import numpy as np

# Simulating the key difference: discrete vs continuous representations

# === Statistical LM: words are indices ===
vocab = {"the": 0, "cat": 1, "dog": 2, "sat": 3, "ran": 4, "mat": 5}

# Bigram counts (sparse matrix)
bigram_counts = np.zeros((len(vocab), len(vocab)))
bigram_counts[0, 1] = 5   # the -> cat
bigram_counts[0, 2] = 3   # the -> dog
bigram_counts[1, 3] = 4   # cat -> sat
bigram_counts[2, 4] = 2   # dog -> ran

# "cat -> ran" is ZERO! No generalization possible.
print("Statistical LM:")
print(f"  P(sat|cat) = {bigram_counts[1,3] / bigram_counts[1,:].sum():.3f}")
print(f"  P(ran|cat) = {bigram_counts[1,4] / max(bigram_counts[1,:].sum(), 1):.3f}")
print(f"  P(sat|dog) = {bigram_counts[2,3] / max(bigram_counts[2,:].sum(), 1):.3f}")

# === Neural LM: words are vectors ===
# Learned embeddings (2D for illustration)
embeddings = {
    "cat": np.array([0.8, 0.2]),   # Similar to dog
    "dog": np.array([0.7, 0.3]),   # Similar to cat
    "sat": np.array([-0.5, 0.9]),  # Similar to ran
    "ran": np.array([-0.4, 0.8]),  # Similar to sat
}

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\\nNeural LM (embedding similarities):")
print(f"  sim(cat, dog) = {cosine_sim(embeddings['cat'], embeddings['dog']):.3f}")
print(f"  sim(sat, ran) = {cosine_sim(embeddings['sat'], embeddings['ran']):.3f}")
print("  -> Because cat ~ dog and sat ~ ran,")
print("     P(ran|cat) > 0 even if never seen in training!")

# Evolution of LM architectures
print("\\nLM Architecture Evolution:")
timeline = [
    ("1980s", "N-gram models",           "Count-based, smoothing"),
    ("2003",  "Bengio's NNLM",           "First neural LM, feedforward"),
    ("2013",  "Word2Vec",                 "Efficient word embeddings"),
    ("2015",  "LSTM/GRU LMs",            "Recurrent, variable context"),
    ("2017",  "Transformer",             "Self-attention, parallel"),
    ("2018",  "GPT / BERT",              "Pre-training + fine-tuning"),
    ("2020",  "GPT-3 (175B)",            "In-context learning emerges"),
    ("2023",  "GPT-4, LLaMA, Claude",    "Reasoning, instruction-following"),
]
for year, model, desc in timeline:
    print(f"  {year}: {model:<25s} {desc}")`,id:"code-lm-comparison"}),e.jsx(l,{type:"intuition",title:"The Curse of Dimensionality, Solved",content:"Bengio (2003) identified the core insight: n-gram models suffer from the curse of dimensionality because they need to see every possible context. Neural models map words to a continuous space where similar contexts produce similar predictions. A model that learns from 'the cat sat' can generalize to 'the kitten sat' because the representations are close.",id:"note-curse"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Key Differences"}),e.jsx("div",{className:"overflow-x-auto",children:e.jsxs("table",{className:"w-full text-sm text-gray-700 dark:text-gray-300",children:[e.jsx("thead",{children:e.jsxs("tr",{className:"border-b border-gray-300 dark:border-gray-600",children:[e.jsx("th",{className:"px-4 py-2 text-left",children:"Aspect"}),e.jsx("th",{className:"px-4 py-2 text-left",children:"Statistical LM"}),e.jsx("th",{className:"px-4 py-2 text-left",children:"Neural LM"})]})}),e.jsxs("tbody",{className:"divide-y divide-gray-200 dark:divide-gray-700",children:[e.jsxs("tr",{children:[e.jsx("td",{className:"px-4 py-2 font-medium",children:"Representation"}),e.jsx("td",{className:"px-4 py-2",children:"Discrete counts"}),e.jsx("td",{className:"px-4 py-2",children:"Dense embeddings"})]}),e.jsxs("tr",{children:[e.jsx("td",{className:"px-4 py-2 font-medium",children:"Generalization"}),e.jsx("td",{className:"px-4 py-2",children:"No similarity notion"}),e.jsx("td",{className:"px-4 py-2",children:"Similar words share parameters"})]}),e.jsxs("tr",{children:[e.jsx("td",{className:"px-4 py-2 font-medium",children:"Context"}),e.jsx("td",{className:"px-4 py-2",children:"Fixed n-gram window"}),e.jsx("td",{className:"px-4 py-2",children:"Variable (up to context length)"})]}),e.jsxs("tr",{children:[e.jsx("td",{className:"px-4 py-2 font-medium",children:"Parameters"}),e.jsx("td",{className:"px-4 py-2",children:"Millions of counts"}),e.jsx("td",{className:"px-4 py-2",children:"Millions to trillions of weights"})]}),e.jsxs("tr",{children:[e.jsx("td",{className:"px-4 py-2 font-medium",children:"Training"}),e.jsx("td",{className:"px-4 py-2",children:"Simple counting"}),e.jsx("td",{className:"px-4 py-2",children:"Gradient descent (GPU-intensive)"})]})]})]})}),e.jsx(l,{type:"historical",title:"Bengio's 2003 Paper",content:"'A Neural Probabilistic Language Model' by Bengio et al. (2003) is one of the most influential NLP papers. It introduced the idea of learning distributed word representations jointly with a language model. Though it took over a decade for the ideas to fully mature (through Word2Vec, LSTMs, and finally Transformers), this paper laid the conceptual foundation for all modern LLMs.",id:"note-bengio"})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:G},Symbol.toStringTag,{value:"Module"}));function H(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Cleaning and Normalization"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Real-world text is messy. Web scrapes contain HTML tags, social media has irregular spelling, and documents mix encodings. Text cleaning and normalization transform raw text into a consistent format before tokenization or model training. The quality of your data pipeline directly determines the quality of your model."}),e.jsx(c,{title:"Text Normalization",definition:"Text normalization is the process of transforming text into a canonical (standard) form. This includes case folding, Unicode normalization, whitespace standardization, and removal of irrelevant content. The goal is to reduce surface variation while preserving semantic content.",id:"def-normalization"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Common Cleaning Steps"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The specific cleaning steps depend on your data source and task, but common operations include:"}),e.jsx(d,{title:"text_cleaning.py",code:`import re
import unicodedata
import html

def clean_text(text):
    """Comprehensive text cleaning pipeline."""
    # 1. Decode HTML entities
    text = html.unescape(text)  # &amp; -> &, &lt; -> <

    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # 3. Unicode normalization (NFKC: compatibility decomposition + canonical composition)
    text = unicodedata.normalize('NFKC', text)

    # 4. Replace common Unicode variants
    text = text.replace('‘', "'").replace('’', "'")  # Smart quotes
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('—', '--')  # Em dash
    text = text.replace(' ', ' ')   # Non-breaking space

    # 5. Normalize whitespace
    text = re.sub(r's+', ' ', text).strip()

    # 6. Remove control characters (keep newlines optionally)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cc' or c in '
	')

    return text

# Example: messy web-scraped text
raw = """<p>This is &ldquo;great&rdquo; content!&nbsp;&nbsp;
   <b>Bold</b> claims   with weird—spacing.</p>
   Visit us at <a href="http://example.com">our site</a>."""

cleaned = clean_text(raw)
print(f"Raw:     {repr(raw[:80])}")
print(f"Cleaned: {repr(cleaned)}")

# Case normalization (context-dependent!)
text = "Apple released the iPhone in San Francisco"
print(f"\\nLowered: {text.lower()}")
# Warning: lowercasing loses entity info ("Apple" company vs "apple" fruit)`,id:"code-cleaning"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Unicode Normalization Forms"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Unicode provides four normalization forms. The choice matters because visually identical strings may have different byte representations:"}),e.jsx(d,{title:"unicode_normalization.py",code:`import unicodedata

# The accented 'e' has two representations
# Composed: single code point U+00E9
e_composed = 'é'  # e with acute accent
# Decomposed: base 'e' + combining acute accent
e_decomposed = 'é'

print(f"Composed:   '{e_composed}' (len={len(e_composed)}, bytes={e_composed.encode('utf-8').hex()})")
print(f"Decomposed: '{e_decomposed}' (len={len(e_decomposed)}, bytes={e_decomposed.encode('utf-8').hex()})")
print(f"Look same?  {e_composed} == {e_decomposed}? {e_composed == e_decomposed}")

# NFC (Canonical Decomposition + Canonical Composition) - recommended default
nfc = unicodedata.normalize('NFC', e_decomposed)
print(f"\\nAfter NFC:  '{nfc}' == composed? {nfc == e_composed}")

# NFKC also converts compatibility characters
text = 'ﬁ'  # fi ligature
print(f"\\nLigature: '{text}' -> NFKC: '{unicodedata.normalize('NFKC', text)}'")

# Practical example: searching for "cafe" should match "café"
def normalize_for_search(text):
    return unicodedata.normalize('NFKC', text).lower()

query = "cafe"
documents = ["café latte", "café mocha", "cafe americano"]
for doc in documents:
    match = normalize_for_search(query) in normalize_for_search(doc)
    print(f"  '{query}' in '{doc}': {match}")`,id:"code-unicode-norm"}),e.jsx(p,{title:"Cleaning Pipeline Order Matters",problem:"Given HTML text: '<p>The caf&eacute; is GREAT!!!</p>', apply cleaning steps in the correct order.",steps:[{formula:"Step 1: HTML unescape -> '<p>The caf\\u00e9 is GREAT!!!</p>'",explanation:"Decode HTML entities first, before removing tags."},{formula:"Step 2: Strip HTML tags -> 'The caf\\u00e9 is GREAT!!!'",explanation:"Remove markup after unescaping."},{formula:"Step 3: Unicode normalize (NFC) -> 'The caf\\u00e9 is GREAT!!!'",explanation:"Normalize Unicode representations."},{formula:"Step 4: Lowercase (if needed) -> 'the caf\\u00e9 is great!!!'",explanation:"Case folding depends on your task."}],id:"example-cleaning-order"}),e.jsx(u,{title:"Do Not Over-Clean",content:"Aggressive cleaning can destroy useful signal. For LLM training, preserving case, punctuation, and formatting is often important because the model should learn to handle real text. Modern LLMs trained on lightly-cleaned data outperform those trained on heavily normalized data. Clean just enough to remove true noise (broken HTML, encoding errors) without removing linguistic variation.",id:"warning-over-clean"}),e.jsx(l,{type:"tip",title:"Language-Specific Cleaning",content:"Different languages need different normalization. Chinese and Japanese text has no spaces between words and needs segmentation. Arabic requires handling diacritics. German has compound words. Hindi uses Devanagari conjuncts. Always consider the target language when designing your cleaning pipeline.",id:"note-language-specific"})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:H},Symbol.toStringTag,{value:"Module"}));function X(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Stemming and Lemmatization"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:'Words appear in many inflected forms: "running", "runs", "ran" all relate to the concept of "run". Stemming and lemmatization reduce words to a base form, shrinking the effective vocabulary and grouping related words together. These techniques are fundamental to information retrieval and classical NLP.'}),e.jsx(c,{title:"Stemming",definition:"Stemming is a heuristic process that chops off word suffixes to produce a stem. The stem may not be a valid word. For example, the Porter stemmer reduces 'running' to 'run', 'studies' to 'studi', and 'university' to 'univers'.",id:"def-stemming"}),e.jsx(c,{title:"Lemmatization",definition:"Lemmatization uses morphological analysis and vocabulary lookup to reduce a word to its dictionary form (lemma). Unlike stemming, the result is always a valid word. For example, 'better' lemmatizes to 'good', 'mice' to 'mouse', and 'ran' to 'run'.",id:"def-lemmatization"}),e.jsx(p,{title:"Stemming vs Lemmatization",problem:"Compare stemming and lemmatization for the words: 'studies', 'studying', 'better', 'wolves'",steps:[{formula:"'studies' -> stem: 'studi', lemma: 'study'",explanation:"The stemmer applies a crude suffix rule; the lemmatizer finds the dictionary form."},{formula:"'studying' -> stem: 'studi', lemma: 'study'",explanation:"Both reduce to the base, but the stem is not a real word."},{formula:"'better' -> stem: 'better', lemma: 'good'",explanation:"Stemmers cannot handle irregular forms; lemmatizers use morphological rules."},{formula:"'wolves' -> stem: 'wolv', lemma: 'wolf'",explanation:"Lemmatization correctly handles irregular plurals."}],id:"example-stem-vs-lemma"}),e.jsx(d,{title:"stemming_lemmatization.py",code:`import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Download required data (run once)
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

porter = PorterStemmer()
snowball = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

words = [
    'running', 'runs', 'ran',
    'studies', 'studying', 'studied',
    'better', 'best', 'good',
    'wolves', 'mice', 'geese',
    'happily', 'happiness', 'unhappy',
    'organization', 'organizing', 'organized',
]

print(f"{'Word':<16} {'Porter':<16} {'Snowball':<16} {'Lemma (v)':<16} {'Lemma (n)':<16}")
print("-" * 80)
for word in words:
    p = porter.stem(word)
    s = snowball.stem(word)
    lv = lemmatizer.lemmatize(word, pos='v')  # As verb
    ln = lemmatizer.lemmatize(word, pos='n')  # As noun
    print(f"{word:<16} {p:<16} {s:<16} {lv:<16} {ln:<16}")

# Why POS matters for lemmatization
print("\\nPOS-aware lemmatization:")
print(f"  'saw' as noun: {lemmatizer.lemmatize('saw', 'n')}")   # saw (tool)
print(f"  'saw' as verb: {lemmatizer.lemmatize('saw', 'v')}")   # see`,id:"code-stemming"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Stemming Algorithms"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The most common stemmers apply cascading rules to strip suffixes:"}),e.jsxs("ul",{className:"ml-6 list-disc space-y-2 text-gray-700 dark:text-gray-300",children:[e.jsxs("li",{children:[e.jsx("strong",{children:"Porter Stemmer (1980)"})," - The classic algorithm with 5 phases of suffix-stripping rules. Fast but aggressive."]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Snowball Stemmer"})," - Martin Porter's improved version with better rules and multi-language support."]}),e.jsxs("li",{children:[e.jsx("strong",{children:"Lancaster Stemmer"})," - More aggressive than Porter, producing shorter stems."]})]}),e.jsx(d,{title:"stemming_in_search.py",code:`from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stemmer = PorterStemmer()

def stem_tokenizer(text):
    """Tokenize and stem for use in TF-IDF."""
    tokens = text.lower().split()
    return [stemmer.stem(t) for t in tokens]

documents = [
    "The dogs are running in the park",
    "A dog runs quickly through parks",
    "Cats sleep on comfortable beds",
    "The cat is sleeping peacefully on a bed",
]

# TF-IDF with stemming
tfidf_stem = TfidfVectorizer(tokenizer=stem_tokenizer)
X_stem = tfidf_stem.fit_transform(documents)

# TF-IDF without stemming
tfidf_raw = TfidfVectorizer()
X_raw = tfidf_raw.fit_transform(documents)

print("Cosine similarity D1 vs D2 (related: dogs running/runs):")
print(f"  Without stemming: {cosine_similarity(X_raw[0:1], X_raw[1:2])[0][0]:.3f}")
print(f"  With stemming:    {cosine_similarity(X_stem[0:1], X_stem[1:2])[0][0]:.3f}")

print("\\nCosine similarity D3 vs D4 (related: cats sleeping):")
print(f"  Without stemming: {cosine_similarity(X_raw[2:3], X_raw[3:4])[0][0]:.3f}")
print(f"  With stemming:    {cosine_similarity(X_stem[2:3], X_stem[3:4])[0][0]:.3f}")
# Stemming improves similarity for semantically related documents!`,id:"code-stem-search"}),e.jsx(u,{title:"Stemming Errors",content:"Stemmers can both over-stem (merging unrelated words: 'university' and 'universe' both stem to 'univers') and under-stem (failing to merge related words: 'alumnus' and 'alumni' produce different stems). These errors can hurt precision in search and classification tasks.",id:"warning-stem-errors"}),e.jsx(l,{type:"note",title:"Modern LLMs and Stemming",content:"Subword tokenizers (BPE, WordPiece) largely eliminate the need for explicit stemming or lemmatization. They naturally capture morphological structure: 'running' might be tokenized as 'run' + 'ning', letting the model learn the relationship implicitly. However, stemming remains valuable for lightweight search indices, traditional IR systems, and low-resource languages.",id:"note-modern-stemming"})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:X},Symbol.toStringTag,{value:"Module"}));function K(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Stop Words and Vocabulary"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:'Not all words carry equal information. Function words like "the", "is", and "of" appear frequently but contribute little to document meaning. Managing vocabulary size and filtering uninformative tokens is a key preprocessing decision that affects both model performance and computational cost.'}),e.jsx(c,{title:"Stop Words",definition:"Stop words are high-frequency, low-information words that are often removed during text preprocessing. They include articles (the, a), prepositions (in, on, at), conjunctions (and, but, or), and common verbs (is, are, was). Different stop word lists exist for different languages and tasks.",id:"def-stopwords"}),e.jsx(d,{title:"stopwords_exploration.py",code:`import nltk
from collections import Counter

# nltk.download('stopwords')  # Run once
from nltk.corpus import stopwords

# English stop words
en_stops = set(stopwords.words('english'))
print(f"NLTK English stop words: {len(en_stops)}")
print(f"Examples: {sorted(list(en_stops))[:20]}")

# Demonstrate the impact of stop word removal
text = """The transformer architecture has fundamentally changed the field
of natural language processing. It uses self-attention mechanisms
to process all tokens in a sequence simultaneously, which is much
more efficient than the recurrent approaches that came before it."""

tokens = text.lower().split()
filtered = [t for t in tokens if t not in en_stops]

print(f"\\nOriginal:  {len(tokens)} tokens")
print(f"Filtered:  {len(filtered)} tokens ({len(tokens)-len(filtered)} removed)")
print(f"\\nOriginal first 15:  {tokens[:15]}")
print(f"Filtered first 15:  {filtered[:15]}")

# Zipf's law: word frequency follows a power law
all_tokens = text.lower().split()
freq = Counter(all_tokens)
print("\\nTop 10 words by frequency:")
for word, count in freq.most_common(10):
    is_stop = "STOP" if word in en_stops else ""
    print(f"  {word:<15} {count:>3}  {is_stop}")
# Most frequent words are stop words!`,id:"code-stopwords"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Zipf's Law and Vocabulary"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Word frequencies in natural language follow Zipf's law: the frequency of a word is inversely proportional to its rank in the frequency table."}),e.jsxs("div",{className:"my-4",children:[e.jsx(r.BlockMath,{math:"f(r) \\propto \\frac{1}{r^s}"}),e.jsxs("p",{className:"text-center text-sm text-gray-500 dark:text-gray-400",children:["where ",e.jsx(r.InlineMath,{math:"r"})," is the rank and ",e.jsx(r.InlineMath,{math:"s \\approx 1"})," for English text."]})]}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"This means a small number of words account for most of the text (stop words), while the vast majority of unique words are rare. This long tail creates a fundamental tension in vocabulary design."}),e.jsx(c,{title:"Vocabulary",definition:"A vocabulary $V$ is the set of unique tokens a model can represent. Tokens outside the vocabulary are unknown (OOV). Vocabulary size is a key hyperparameter: too small and information is lost, too large and the model becomes inefficient.",id:"def-vocabulary"}),e.jsx(d,{title:"vocabulary_management.py",code:`from collections import Counter
import numpy as np

# Simulating vocabulary statistics on a corpus
# (Using word frequencies from a hypothetical corpus)
np.random.seed(42)
# Generate Zipfian word frequencies
vocab_size = 10000
ranks = np.arange(1, vocab_size + 1)
frequencies = (1.0 / ranks) * 100000  # Zipf's law
frequencies = frequencies.astype(int)

total_tokens = frequencies.sum()
cumulative = np.cumsum(frequencies) / total_tokens

# How many words cover X% of all tokens?
for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
    n_words = np.searchsorted(cumulative, threshold) + 1
    print(f"  {threshold*100:.0f}% coverage: {n_words:,} words "
          f"({n_words/vocab_size*100:.1f}% of vocabulary)")

# Vocabulary pruning strategies
print("\\nVocabulary reduction strategies:")
strategies = {
    "Min frequency = 2":  sum(1 for f in frequencies if f >= 2),
    "Min frequency = 5":  sum(1 for f in frequencies if f >= 5),
    "Min frequency = 10": sum(1 for f in frequencies if f >= 10),
    "Top 5,000 words":    5000,
    "Top 1,000 words":    1000,
}
for name, size in strategies.items():
    print(f"  {name:<25} -> vocab size: {size:,}")

# Special tokens in modern LLM vocabularies
special_tokens = {
    "<pad>":   "Padding for batch processing",
    "<unk>":   "Unknown/OOV token",
    "<bos>":   "Beginning of sequence",
    "<eos>":   "End of sequence",
    "<mask>":  "Masked position (BERT-style)",
    "<sep>":   "Separator between segments",
}
print("\\nSpecial tokens in LLM vocabularies:")
for token, desc in special_tokens.items():
    print(f"  {token:<10} {desc}")`,id:"code-vocabulary"}),e.jsx(p,{title:"When to Remove Stop Words",problem:"Should you remove stop words for a sentiment analysis task on product reviews?",steps:[{formula:'Consider: "This is not good" vs "This is good"',explanation:'"not" is a stop word, but removing it flips the sentiment entirely!'},{formula:'Consider: "I could not be happier"',explanation:'"could not be" are all stop words, but they carry critical sentiment information.'},{formula:"Decision: Do NOT remove stop words for sentiment analysis",explanation:"Negation words and function words carry grammatical meaning essential for sentiment."}],id:"example-stopword-decision"}),e.jsx(u,{title:"Stop Word Removal Is Task-Dependent",content:"Stop word removal helps for topic modeling, keyword extraction, and search (TF-IDF). But it hurts for tasks that depend on syntax: sentiment analysis, question answering, and machine translation. For LLM training, stop words are never removed because the model needs to generate fluent text with proper grammar.",id:"warning-task-dependent"}),e.jsx(l,{type:"tip",title:"Frequency-Based vs. List-Based Filtering",content:"Instead of using a fixed stop word list, consider frequency-based filtering: remove words that appear in more than X% of documents (too common) or fewer than Y documents (too rare). scikit-learn's TfidfVectorizer supports this with max_df and min_df parameters, providing data-driven vocabulary selection.",id:"note-frequency-filtering"})]})}const ue=Object.freeze(Object.defineProperty({__proto__:null,default:K},Symbol.toStringTag,{value:"Module"}));function Y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Data Pipelines for NLP"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Building an NLP system requires more than just a model. You need a robust data pipeline that ingests raw text, cleans it, transforms it into model-ready features, and handles edge cases gracefully. A well-designed pipeline is reproducible, efficient, and modular."}),e.jsx(c,{title:"NLP Data Pipeline",definition:"An NLP data pipeline is a sequence of processing stages that transforms raw text data into a format suitable for model training or inference. Typical stages include: data collection, cleaning, normalization, tokenization, encoding, batching, and optional augmentation.",id:"def-pipeline"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"A Complete Preprocessing Pipeline"}),e.jsx(d,{title:"nlp_pipeline.py",code:`import re
import unicodedata
from collections import Counter

class TextPipeline:
    """A modular NLP preprocessing pipeline."""

    def __init__(self, steps=None):
        self.steps = steps or [
            self.clean_html,
            self.normalize_unicode,
            self.normalize_whitespace,
            self.lowercase,
            self.tokenize,
        ]

    def __call__(self, text):
        result = text
        for step in self.steps:
            result = step(result)
        return result

    @staticmethod
    def clean_html(text):
        """Remove HTML tags and decode entities."""
        import html
        text = html.unescape(text)
        return re.sub(r'<[^>]+>', ' ', text)

    @staticmethod
    def normalize_unicode(text):
        """NFKC normalize and remove control characters."""
        text = unicodedata.normalize('NFKC', text)
        return ''.join(c for c in text if unicodedata.category(c) != 'Cc' or c == '
')

    @staticmethod
    def normalize_whitespace(text):
        return re.sub(r's+', ' ', text).strip()

    @staticmethod
    def lowercase(text):
        return text.lower()

    @staticmethod
    def tokenize(text):
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r"\bw+\b", text)

# Use the pipeline
pipeline = TextPipeline()
raw = "<p>The Transformer &amp; BERT models are GREAT!! 🚀</p>"
tokens = pipeline(raw)
print(f"Raw:    {raw}")
print(f"Tokens: {tokens}")

# Custom pipeline for different tasks
search_pipeline = TextPipeline(steps=[
    TextPipeline.normalize_unicode,
    TextPipeline.normalize_whitespace,
    TextPipeline.lowercase,
    TextPipeline.tokenize,
])
print(f"\\nSearch pipeline: {search_pipeline('  Find BERT   Models  ')}")`,id:"code-pipeline"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Batching and Padding"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Neural networks process data in batches for efficiency. Since text sequences have varying lengths, they must be padded to a uniform length within each batch."}),e.jsx(d,{title:"batching_padding.py",code:`import numpy as np

def create_batches(sequences, batch_size=3, pad_token=0, max_len=None):
    """
    Create padded batches from variable-length sequences.
    Returns batches of (padded_sequences, attention_masks, lengths).
    """
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        lengths = [len(seq) for seq in batch]

        # Pad to the max length in this batch (or global max_len)
        pad_len = max_len or max(lengths)
        padded = np.full((len(batch), pad_len), pad_token, dtype=np.int64)
        attention_mask = np.zeros((len(batch), pad_len), dtype=np.int64)

        for j, seq in enumerate(batch):
            seq_len = min(len(seq), pad_len)
            padded[j, :seq_len] = seq[:seq_len]
            attention_mask[j, :seq_len] = 1

        batches.append({
            'input_ids': padded,
            'attention_mask': attention_mask,
            'lengths': lengths,
        })
    return batches

# Simulate tokenized sequences of different lengths
sequences = [
    [101, 2023, 2003, 1037, 3231, 102],         # 6 tokens
    [101, 2312, 6251, 102],                       # 4 tokens
    [101, 1996, 4937, 4540, 2006, 1996, 13523, 102],  # 8 tokens
    [101, 7592, 102],                              # 3 tokens
    [101, 2028, 2062, 6251, 2182, 102],           # 6 tokens
]

batches = create_batches(sequences, batch_size=3)
for i, batch in enumerate(batches):
    print(f"Batch {i+1}:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Attention mask:\\n{batch['attention_mask']}")
    print(f"  Lengths: {batch['lengths']}\\n")`,id:"code-batching"}),e.jsx(p,{title:"Pipeline Design Decisions",problem:"You are building a pipeline to prepare data for fine-tuning a BERT model on movie review sentiment. What preprocessing steps should you include?",steps:[{formula:"Step 1: HTML cleaning",explanation:"Movie reviews from the web may contain HTML tags, entities, and formatting."},{formula:"Step 2: Unicode normalization (NFKC)",explanation:"Standardize character representations. Do not remove emojis -- they carry sentiment."},{formula:"Step 3: BERT tokenizer (WordPiece)",explanation:"Use the pre-trained tokenizer that matches your BERT model. Do NOT lowercase if using cased BERT."},{formula:"Step 4: Truncation to 512 tokens",explanation:"BERT has a maximum sequence length of 512. Truncate or split longer reviews."},{formula:"Step 5: Add special tokens [CLS] and [SEP]",explanation:"BERT requires these boundary markers."}],id:"example-pipeline-design"}),e.jsx(l,{type:"tip",title:"Data Quality Over Quantity",content:"The LLaMA paper (Touvron et al., 2023) showed that a smaller model trained on high-quality, well-curated data can outperform larger models trained on raw web scrapes. Data deduplication, quality filtering (using perplexity-based heuristics), and domain balancing are now considered as important as model architecture.",id:"note-data-quality"}),e.jsx(u,{title:"Reproducibility",content:"Always version your preprocessing pipeline alongside your model. A change in tokenizer version, Unicode normalization form, or cleaning rules can silently change your data distribution and invalidate comparisons. Use deterministic processing and log every transformation step.",id:"warning-reproducibility"}),e.jsx(l,{type:"note",title:"Modern Data Pipeline Tools",content:"Production NLP pipelines use tools like Hugging Face Datasets (memory-mapped, lazy processing), Apache Beam (distributed processing), and Spark NLP. For LLM pre-training, specialized tools like The Pile's data pipeline, RedPajama, and Dolma handle terabytes of text with deduplication, quality scoring, and PII removal.",id:"note-tools"})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:Y},Symbol.toStringTag,{value:"Module"}));export{c as D,p as E,l as N,d as P,j as T,u as W,te as a,ne as b,ae as c,ie as d,se as e,re as f,oe as g,le as h,de as i,ce as j,me as k,pe as l,he as m,ue as n,ge as o,ee as s};
