# AI-course
A concise two-column blog theme for Jekyll.


| 日期 | 标题 | 封面 | 时长 | 视频（播放数） |
| --: | -- | -- | --: | -- |
| 1/11/23 | 多模态论文串讲·下 |  <img src="imgs/multimodal-2.jpg" width="200px"/> | 1:03:29 | [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1fA411Z772)](https://www.bilibili.com/video/BV1fA411Z772) <br />[![](https://img.shields.io/youtube/views/S1le41J76lQ?style=social)](https://youtu.be/S1le41J76lQ) |
| 12/29/22 | [Instruct GPT](https://arxiv.org/pdf/2203.02155.pdf) | <img src="imgs/instruct-gpt.jpg" width="200px"/> | 1:07:10 | [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1hd4y187CR)](https://www.bilibili.com/video/BV1hd4y187CR) <br />[![](https://img.shields.io/youtube/views/zfIGAwD1jOQ?style=social)](https://youtu.be/zfIGAwD1jOQ) |
| 12/19/22 | [Neural Corpus Indexer](https://arxiv.org/pdf/2206.02743.pdf) 文档检索 | <img src="imgs/nci.jpg" width="200px"/> | 55:47 | [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1Se411w7Sn)](https://www.bilibili.com/video/BV1Se411w7Sn) <br />[![](https://img.shields.io/youtube/views/QRffZMSGJyU?style=social)](https://youtu.be/QRffZMSGJyU) |


### mermaid



```mermaid
  flowchart LR;
      A[AIoT]-->B{Internet of Things} & T{Artifical Intellegence};
      classDef green color:#022e1f,fill:#00f500;
      classDef red color:#022e1f,fill:#f11111;
      classDef white color:#022e1f,fill:#fff;
      classDef black color:#fff,fill:#000;
      B-->Z[edge computing]:::green;
      
      Z-->S[sparse matrix] & Q[numerical method];
      C-->U[I choose recaptcha.]:::green;
      U--Views-->Q["echo CIMC_JS('recaptcha');\n echo CIMC_HTML(['captcha_name'=>'recaptcha']);"]:::green;
      U--Controller-->W["CIMC_RULE('recaptcha');"]:::green;
      
      C-->I[I choose arcaptcha.]:::white;
      I--Views-->O["echo CIMC_JS('arcaptcha');\n echo CIMC_HTML(['captcha_name'=>'arcaptcha']);"]:::white;
      I--Controller-->P["CIMC_RULE('arcaptcha');"]:::white;
      
      C-->X[I choose bibot.]:::red;
      X--Views-->V["echo CIMC_JS('bibot');\n echo CIMC_HTML(['captcha_name'=>'bibot']);"]:::red;
      X--Controller-->N["CIMC_RULE('bibot');"]:::red;
      
      B--NO-->D[How to use?]:::black;
      D---Views:::black-->F["echo CIMC_JS('randomcaptcha');\n echo CIMC_HTML(['captcha_name'=>'randomcaptcha']);"]:::black; 
      D---Controller:::black-->T["CIMC_RULE('archaptcha,recaptcha,bibot');"]:::black;
```


```mermaid
sequenceDiagram
    participant Alice
    participant John
    links Alice: {"Dashboard": "https://dashboard.contoso.com/alice", "Wiki": "https://wiki.contoso.com/alice"}
    links John: {"Dashboard": "https://dashboard.contoso.com/john", "Wiki": "https://wiki.contoso.com/john"}
    Alice->>John: Hello John, how are you?
    John-->>Alice: Great!
    Alice-)John: See you later!
```

```mermaid
  flowchart LR;
      A[CI MULTI CHAPTCHA]-->B{Select captcha service by developer?};
      classDef green color:#022e1f,fill:#00f500;
      classDef red color:#022e1f,fill:#f11111;
      classDef white color:#022e1f,fill:#fff;
      classDef black color:#fff,fill:#000;
      B--YES-->C[How to use?]:::green;
      
      C-->U[I choose recaptcha.]:::green;
      U--Views-->Q["echo CIMC_JS('recaptcha');\n echo CIMC_HTML(['captcha_name'=>'recaptcha']);"]:::green;
      U--Controller-->W["CIMC_RULE('recaptcha');"]:::green;
      
      C-->I[I choose arcaptcha.]:::white;
      I--Views-->O["echo CIMC_JS('arcaptcha');\n echo CIMC_HTML(['captcha_name'=>'arcaptcha']);"]:::white;
      I--Controller-->P["CIMC_RULE('arcaptcha');"]:::white;
      
      C-->X[I choose bibot.]:::red;
      X--Views-->V["echo CIMC_JS('bibot');\n echo CIMC_HTML(['captcha_name'=>'bibot']);"]:::red;
      X--Controller-->N["CIMC_RULE('bibot');"]:::red;
      
      B--NO-->D[How to use?]:::black;
      D---Views:::black-->F["echo CIMC_JS('randomcaptcha');\n echo CIMC_HTML(['captcha_name'=>'randomcaptcha']);"]:::black; 
      D---Controller:::black-->T["CIMC_RULE('archaptcha,recaptcha,bibot');"]:::black;
```

```mermaid
flowchart LR;
    A-->B;
    B-->C;
    C-->D;
    click A callback "Tooltip for a callback"
    click B "http://www.github.com" "This is a tooltip for a link"
    click A call callback() "Tooltip for a callback"
    click B href "http://www.github.com" "This is a tooltip for a link"
```

<details>
    <summary>Toggle Switch</summary>
    Foldable Content[enter image description here][1]
</details>
	

