(function(e){function t(t){for(var a,o,i=t[0],c=t[1],l=t[2],u=0,d=[];u<i.length;u++)o=i[u],Object.prototype.hasOwnProperty.call(r,o)&&r[o]&&d.push(r[o][0]),r[o]=0;for(a in c)Object.prototype.hasOwnProperty.call(c,a)&&(e[a]=c[a]);p&&p(t);while(d.length)d.shift()();return s.push.apply(s,l||[]),n()}function n(){for(var e,t=0;t<s.length;t++){for(var n=s[t],a=!0,o=1;o<n.length;o++){var i=n[o];0!==r[i]&&(a=!1)}a&&(s.splice(t--,1),e=c(c.s=n[0]))}return e}var a={},o={app:0},r={app:0},s=[];function i(e){return c.p+"js/"+({about:"about"}[e]||e)+"."+{about:"a84c2c04","chunk-2d0cc7f8":"2829c690","chunk-34e487bb":"542c7ca3","chunk-968b55d2":"0d2804d1"}[e]+".js"}function c(t){if(a[t])return a[t].exports;var n=a[t]={i:t,l:!1,exports:{}};return e[t].call(n.exports,n,n.exports,c),n.l=!0,n.exports}c.e=function(e){var t=[],n={about:1,"chunk-34e487bb":1,"chunk-968b55d2":1};o[e]?t.push(o[e]):0!==o[e]&&n[e]&&t.push(o[e]=new Promise((function(t,n){for(var a="css/"+({about:"about"}[e]||e)+"."+{about:"97652485","chunk-2d0cc7f8":"31d6cfe0","chunk-34e487bb":"ab3c6abd","chunk-968b55d2":"bbb88b27"}[e]+".css",r=c.p+a,s=document.getElementsByTagName("link"),i=0;i<s.length;i++){var l=s[i],u=l.getAttribute("data-href")||l.getAttribute("href");if("stylesheet"===l.rel&&(u===a||u===r))return t()}var d=document.getElementsByTagName("style");for(i=0;i<d.length;i++){l=d[i],u=l.getAttribute("data-href");if(u===a||u===r)return t()}var p=document.createElement("link");p.rel="stylesheet",p.type="text/css",p.onload=t,p.onerror=function(t){var a=t&&t.target&&t.target.src||r,s=new Error("Loading CSS chunk "+e+" failed.\n("+a+")");s.code="CSS_CHUNK_LOAD_FAILED",s.request=a,delete o[e],p.parentNode.removeChild(p),n(s)},p.href=r;var f=document.getElementsByTagName("head")[0];f.appendChild(p)})).then((function(){o[e]=0})));var a=r[e];if(0!==a)if(a)t.push(a[2]);else{var s=new Promise((function(t,n){a=r[e]=[t,n]}));t.push(a[2]=s);var l,u=document.createElement("script");u.charset="utf-8",u.timeout=120,c.nc&&u.setAttribute("nonce",c.nc),u.src=i(e);var d=new Error;l=function(t){u.onerror=u.onload=null,clearTimeout(p);var n=r[e];if(0!==n){if(n){var a=t&&("load"===t.type?"missing":t.type),o=t&&t.target&&t.target.src;d.message="Loading chunk "+e+" failed.\n("+a+": "+o+")",d.name="ChunkLoadError",d.type=a,d.request=o,n[1](d)}r[e]=void 0}};var p=setTimeout((function(){l({type:"timeout",target:u})}),12e4);u.onerror=u.onload=l,document.head.appendChild(u)}return Promise.all(t)},c.m=e,c.c=a,c.d=function(e,t,n){c.o(e,t)||Object.defineProperty(e,t,{enumerable:!0,get:n})},c.r=function(e){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},c.t=function(e,t){if(1&t&&(e=c(e)),8&t)return e;if(4&t&&"object"===typeof e&&e&&e.__esModule)return e;var n=Object.create(null);if(c.r(n),Object.defineProperty(n,"default",{enumerable:!0,value:e}),2&t&&"string"!=typeof e)for(var a in e)c.d(n,a,function(t){return e[t]}.bind(null,a));return n},c.n=function(e){var t=e&&e.__esModule?function(){return e["default"]}:function(){return e};return c.d(t,"a",t),t},c.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},c.p="/",c.oe=function(e){throw console.error(e),e};var l=window["webpackJsonp"]=window["webpackJsonp"]||[],u=l.push.bind(l);l.push=t,l=l.slice();for(var d=0;d<l.length;d++)t(l[d]);var p=u;s.push([0,"chunk-vendors"]),n()})({0:function(e,t,n){e.exports=n("cd49")},"034f":function(e,t,n){"use strict";var a=n("85ec"),o=n.n(a);o.a},"03cc":function(e,t,n){},"655c":function(e,t,n){"use strict";var a=n("03cc"),o=n.n(a);o.a},"85ec":function(e,t,n){},bbd0:function(e,t,n){"use strict";n.d(t,"c",(function(){return a})),n.d(t,"e",(function(){return r})),n.d(t,"d",(function(){return o})),n.d(t,"a",(function(){return i})),n.d(t,"b",(function(){return s}));n("99af"),n("a15b"),n("d3b7"),n("ac1f"),n("25f0"),n("5319"),n("1276");function a(e){var t=new Date-new Date(1e3*e),n=t/1e3,a=Math.floor(n%60),o=n/60,r=Math.floor(o%60),s=o/60,i=Math.floor(s%24),c=s/24,l=Math.floor(c%7),u=Math.floor(c/7);return u>0?"".concat(u," wks, ").concat(l," days"):l>0?"".concat(l," days, ").concat(i," hrs"):i>0?"".concat(i," hrs, ").concat(r," mins"):r>0?"".concat(r," mins, ").concat(a," secs"):"".concat(a," secs")}function o(e){var t=new Date(1e3*e),n=t/1e3,a=Math.floor(n%60),o=n/60,r=Math.floor(o%60),s=o/60,i=Math.floor(s%24),c=s/24,l=Math.floor(c%7),u=Math.floor(c/7);return u>0?"".concat(u," wks, ").concat(l," days"):l>0?"".concat(l," days, ").concat(i," hrs"):i>0?"".concat(i," hrs, ").concat(r," mins"):r>0?"".concat(r," mins, ").concat(a," secs"):"".concat(a," secs")}function r(e){return new Date(1e3*e).toLocaleString()}function s(e){var t=e.toString().split(".");return t[0]=t[0].replace(/\B(?=(\d{3})+(?!\d))/g,","),t.join(".")}function i(e,t){return null==e?"":isNaN(e)?"NaN":e.toPrecision(t)}},cd49:function(e,t,n){"use strict";n.r(t);n("e260"),n("e6cf"),n("cca6"),n("a79d"),n("0cdd");var a=n("2b0e"),o=n("5f5b"),r=n("b1e0");n("ab8b"),n("2dd8");a["default"].use(o["a"]),a["default"].use(r["a"]);var s=function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("div",{class:[{collapsed:e.collapsed},{onmobile:e.isOnMobile}],attrs:{id:"main"}},[n("div",[n("b-button",{directives:[{name:"b-toggle",rawName:"v-b-toggle.rootlog",modifiers:{rootlog:!0}}],staticStyle:{outline:"none","box-shadow":"none"},attrs:{variant:"white",size:"sm"},on:{click:function(t){e.logVisible=!e.logVisible}}})],1),n("div",[n("div",{staticClass:"main"},[n("router-view")],1),n("b-sidebar",{attrs:{id:"sidebar-right",title:"piSTAR Lab IDE",right:"","bg-variant":"dark","text-variant":"light",width:"100%","no-header":""},scopedSlots:e._u([{key:"footer",fn:function(t){var a=t.hide;return[n("div",{staticClass:"d-flex bg-dark text-light align-items-center px-3 py-1 bottombar"},[n("b-button",{staticClass:"mr-auto",attrs:{title:"Switch to Lab View",size:"sm",variant:"light"},on:{click:function(t){a(),e.showScroll()}}},[n("i",{staticClass:"fa fa-flask"}),e._v(" "),n("i",{staticClass:"fa fa-arrow-right"})]),n("strong",{staticClass:"mr-auto footerbrand"},[n("img",{attrs:{height:"24px",src:"/pistar_edit_w.png"}}),e._v(" piSTAR IDE "),n("i",{staticClass:"fa fa-laptop-code"})])],1)]}}])},[n("div",{staticClass:"h-100",staticStyle:{width:"100%"}},[n("iframe",{staticClass:"h-100",staticStyle:{width:"100%"},attrs:{id:"ideframe",src:"http://localhost:7781"}})])])],1),n("sidebar-menu",{attrs:{disableHover:!1,width:e.width,menu:e.menu,collapsed:e.collapsed,theme:e.selectedTheme,"show-one-child":!0},on:{"toggle-collapse":e.onToggleCollapse}},[n("div",{attrs:{slot:"header"},slot:"header"},[n("router-link",{attrs:{to:"/"}},[e.collapsed?e._e():n("div",{staticClass:"logo"},[n("img",{attrs:{height:"42px",src:"/pistar_edit_w.png"}})]),e.collapsed?e._e():n("div",{staticClass:"logo"},[e._v("piSTAR Lab")]),e.collapsed?n("div",{staticClass:"logo"},[n("img",{attrs:{height:"24px",src:"/pistar_edit_w.png"}})]):e._e()])],1),n("div",{attrs:{slot:"footer"},slot:"footer"})]),e.isOnMobile&&!e.collapsed?n("div",{staticClass:"sidebar-overlay",on:{click:function(t){e.collapsed=!0}}}):e._e(),n("b-navbar",{staticClass:"bottombar py-1",attrs:{toggleable:"lg",type:"dark",variant:"info",fixed:"bottom",small:"true"}},[n("strong",{staticClass:"ml-auto footerbrand"},[n("img",{attrs:{height:"24px",src:"/pistar_edit_w.png"}}),e._v(" piSTAR Lab "),n("i",{staticClass:"fa fa-flask"})]),n("b-button",{directives:[{name:"b-toggle",rawName:"v-b-toggle.sidebar-right",modifiers:{"sidebar-right":!0}}],staticClass:"ml-auto",attrs:{title:"Switch to IDE View",size:"sm",variant:"light"},on:{click:function(t){return e.hideScroll()}}},[n("i",{staticClass:"fa fa-arrow-left"}),e._v(" "),n("i",{staticClass:"fa fa-laptop-code"})])],1)],1)},i=[],c=n("bc3a"),l=n.n(c),u=n("cf92"),d={name:"App",components:{},data:function(){return{appConfig:u["a"],logdataoutput:"_",jupyterUrl:"NA",rayDashUrl:"",logVisible:!1,menu:[{href:"/task/new/multiagent/",title:"New Session Task",icon:"fas fa-plus"},{title:"",header:!0},{href:"/agent/home",title:"Agents",icon:"fas fa-robot"},{href:"/component_spec/home",title:"Components",icon:"fa fa-cogs"},{href:"/env/home",title:"Environments",icon:"fa fa-gamepad"},{title:"",header:!0},{href:"/task/home",title:"Tasks",icon:"fas fa-stream"},{href:"/session/home",title:"Sessions",icon:"fa fa-cubes"},{title:"",header:!0},{href:"/data_browser",title:"Data Browser",icon:"far fa-folder"},{href:"/plugin/home",title:"Plugins",icon:"fa fa-plug"},{href:"/preferences",title:"Preferences",icon:"fa fa-cog"}],collapsed:!1,themes:[{name:"Default theme",input:""},{name:"White theme",input:"white-theme"}],selectedTheme:"",isOnMobile:!1,es:null,logStream:null,logdata:[],logInit:!1}},mounted:function(){this.onResize(),window.addEventListener("resize",this.onResize)},computed:{},methods:{showScroll:function(){document.body.style.overflow="visible"},hideScroll:function(){document.body.style.overflow="hidden"},onToggleCollapse:function(e){console.log(e),this.collapsed=e},onResize:function(){window.innerWidth<=2600?(this.isOnMobile=!0,this.collapsed=!0):(this.isOnMobile=!1,this.collapsed=!1)}},created:function(){var e=this;Object(u["b"])().then((function(t){e.jupyterUrl="http://localhost:"+t["env"]["JUPYTER_PORT"],e.rayDashUrl="http://localhost:"+t["env"]["RAY_DASHBOARD_PORT"]}))},props:{width:{type:String,default:"200px"}}},p=d,f=(n("034f"),n("2877")),h=Object(f["a"])(p,s,i,!1,null,null,null),m=h.exports,b=n("9483");Object(b["a"])("".concat("/","service-worker.js"),{ready:function(){console.log("App is being served from cache by a service worker.\nFor more details, visit https://goo.gl/AFskqB")},registered:function(){console.log("Service worker has been registered.")},cached:function(){console.log("Content has been cached for offline use.")},updatefound:function(){console.log("New content is downloading.")},updated:function(){console.log("New content is available; please refresh.")},offline:function(){console.log("No internet connection found. App is running in offline mode.")},error:function(e){console.error("Error during service worker registration:",e)}});n("b0c0"),n("d3b7");var g=n("8c4f"),v=function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("div",{staticClass:"widget"},[n("b-breadcrumb",[n("b-breadcrumb-item",[n("i",{staticClass:"fa fa-home"})])],1),n("b-card",{staticClass:"h-100"},[n("b-card-text",[n("p",[e._v("Welcome to piSTAR Lab.")])])],1),n("div",{staticClass:"mt-4"}),n("b-card",{attrs:{variant:"dark",title:"Powered by"}},[n("b-card-text",[n("b-link",{staticStyle:{color:"black"},attrs:{href:"https://pytortch.org"}},[n("img",{attrs:{height:"50px",src:"https://pytorch.org/assets/images/logo-icon.svg",alt:"PyTorch"}}),e._v(" PyTorch ")]),n("b-link",{staticClass:"ml-3",attrs:{href:"https://ray.io"}},[n("img",{attrs:{height:"50px",src:"https://raw.githubusercontent.com/ray-project/ray/master/doc/source/images/ray_header_logo.png",alt:"Ray.io"}})]),n("b-link",{staticClass:"ml-3",staticStyle:{color:"black"},attrs:{href:"https://vuejs.org/"}},[n("img",{attrs:{height:"50px",src:"https://vuejs.org/images/logo.png?_sw-precache=cf23526f451784ff137f161b8fe18d5a",alt:"Vue.js"}}),e._v(" Vue.js ")]),n("b-link",{staticClass:"ml-3",attrs:{href:"https://vuejs.org/"}},[n("svg",{attrs:{height:"50px",xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 936 232"}},[n("path",{attrs:{d:"M667.21,90.58c-13.76,0-23.58,4.7-28.4,13.6L636.22,109V92.9H613.83v97.86h23.55V132.54c0-13.91,7.56-21.89,20.73-21.89,12.56,0,19.76,7.77,19.76,21.31v58.8h23.56v-63C701.43,104.46,688.64,90.58,667.21,90.58ZM553,90.58c-27.79,0-45,17.34-45,45.25v13.74c0,26.84,17.41,43.51,45.44,43.51,18.75,0,31.89-6.87,40.16-21L579,163.68c-6.11,8.15-15.87,13.2-25.55,13.2-14.19,0-22.66-8.76-22.66-23.44v-3.89h65.73V133.32c0-26-17.07-42.74-43.5-42.74Zm22.09,43.15H530.71v-2.35c0-16.11,7.91-25,22.27-25,13.83,0,22.09,8.76,22.09,23.44ZM935.31,76.79V58.07H853.85V76.79h28.56V172H853.85v18.72h81.46V172H906.74V76.79ZM317.65,55.37c-36.38,0-59,22.67-59,59.18v19.74c0,36.5,22.61,59.18,59,59.18s59-22.68,59-59.18V114.55C376.64,78,354,55.37,317.65,55.37Zm34.66,80.27c0,24.24-12.63,38.14-34.66,38.14S283,159.88,283,135.64V113.19c0-24.24,12.64-38.14,34.66-38.14s34.66,13.9,34.66,38.14Zm98.31-45.06c-12.36,0-23.06,5.12-28.64,13.69l-2.53,3.9V92.9h-22.4V224.43h23.56V176.79l2.52,3.74c5.3,7.86,15.65,12.55,27.69,12.55,20.31,0,40.8-13.27,40.8-42.93V133.51c0-21.37-12.63-42.93-41-42.93ZM468.06,149c0,15.77-9.2,25.57-24,25.57-13.8,0-23.43-10.36-23.43-25.18V134.67c0-15,9.71-25.56,23.63-25.56,14.69,0,23.82,9.79,23.82,25.56ZM766.53,58.08,719,190.76h23.93l9.1-28.44h54.64l.09.28,9,28.16h23.92L792.07,58.07Zm-8.66,85.53,21.44-67.08,21.22,67.08Z"}}),n("path",{attrs:{d:"M212.59,95.12a57.27,57.27,0,0,0-4.92-47.05,58,58,0,0,0-62.4-27.79A57.29,57.29,0,0,0,102.06,1,57.94,57.94,0,0,0,46.79,41.14,57.31,57.31,0,0,0,8.5,68.93a58,58,0,0,0,7.13,67.94,57.31,57.31,0,0,0,4.92,47A58,58,0,0,0,83,211.72,57.31,57.31,0,0,0,126.16,231a57.94,57.94,0,0,0,55.27-40.14,57.3,57.3,0,0,0,38.28-27.79A57.92,57.92,0,0,0,212.59,95.12ZM126.16,216a42.93,42.93,0,0,1-27.58-10c.34-.19,1-.52,1.38-.77l45.8-26.44a7.43,7.43,0,0,0,3.76-6.51V107.7l19.35,11.17a.67.67,0,0,1,.38.54v53.45A43.14,43.14,0,0,1,126.16,216ZM33.57,176.46a43,43,0,0,1-5.15-28.88c.34.21.94.57,1.36.81l45.81,26.45a7.44,7.44,0,0,0,7.52,0L139,142.52v22.34a.67.67,0,0,1-.27.6L92.43,192.18a43.14,43.14,0,0,1-58.86-15.77Zm-12-100A42.92,42.92,0,0,1,44,57.56V112a7.45,7.45,0,0,0,3.76,6.51l55.9,32.28L84.24,162a.68.68,0,0,1-.65.06L37.3,135.33A43.13,43.13,0,0,1,21.53,76.46Zm159,37-55.9-32.28L144,70a.69.69,0,0,1,.65-.06l46.29,26.73a43.1,43.1,0,0,1-6.66,77.76V120a7.44,7.44,0,0,0-3.74-6.54Zm19.27-29c-.34-.21-.94-.57-1.36-.81L152.67,57.2a7.44,7.44,0,0,0-7.52,0L89.25,89.47V67.14a.73.73,0,0,1,.28-.6l46.29-26.72a43.1,43.1,0,0,1,64,44.65ZM78.7,124.3,59.34,113.13a.73.73,0,0,1-.37-.54V59.14A43.09,43.09,0,0,1,129.64,26c-.34.19-.95.52-1.38.77L82.46,53.21a7.45,7.45,0,0,0-3.76,6.51Zm10.51-22.67,24.9-14.38L139,101.63v28.74L114.1,144.75,89.2,130.37Z"}})])])],1)],1)],1)},w=[],k={},y=Object(f["a"])(k,v,w,!1,null,null,null),_=y.exports,S=function(){var e=this,t=e.$createElement;e._self._c;return e._m(0)},C=[function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("div",{staticClass:"widget"},[n("h2",[e._v("Help")])])}],T={name:"Help",components:{}},A=T,L=Object(f["a"])(A,S,C,!1,null,null,null),O=L.exports,x=function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("div",{staticClass:"widget"},[n("b-breadcrumb",[n("b-breadcrumb-item",{attrs:{to:"/"}},[n("i",{staticClass:"fa fa-home"})]),n("b-breadcrumb-item",{attrs:{active:""}},[n("i",{staticClass:"fa fa-stream"}),e._v(" Tasks")])],1),n("b-button-toolbar",[n("b-button-group",{staticClass:"mr-1"},[n("b-button",{attrs:{variant:"outline-primary",to:"/task/specs/",size:"sm"}},[e._v("New Task")])],1),n("b-button-group",[n("b-button",{attrs:{title:"Browse data",variant:"outline-primary",to:"/data_browser/?path=task",size:"sm"}},[n("i",{staticClass:"fa fa-folder"})])],1)],1),n("div",{staticClass:"mt-4"}),n("b-container",{attrs:{fluid:""}},[n("b-row",[n("b-col",[n("b-card",[e.$apollo.queries.allTask.loading?n("div",[e._v("Loading..")]):n("div",[Object.keys(e.taskList).length>0?n("b-card-text",[n("b-form-checkbox-group",{model:{value:e.selected,callback:function(t){e.selected=t},expression:"selected"}},[n("b-table",{attrs:{hover:"","table-busy":"",items:e.taskList,fields:e.fields,dark:!1,outlined:!1},scopedSlots:e._u([{key:"cell(link)",fn:function(t){return[n("router-link",{attrs:{to:"/task/view/"+t.item.ident}},[e._v(e._s(t.item.ident))])]}},{key:"cell(actions)",fn:function(t){return[t.item.status&&"RUNNING"==t.item.status?n("b-button",{attrs:{size:"sm",variant:"danger"},on:{click:function(n){return e.taskControl("STOP",t.item.ident)}}},[e._v("Terminate")]):e._e()]}}],null,!1,3824146954)})],1),n("p",[e._v(e._s(e.error))])],1):n("b-card-text",[e._v("No Items Found ")])],1)])],1)],1)],1)],1)},E=[],P=(n("99af"),n("d81d"),n("8785")),V=(n("bbd0"),n("9530")),j=n.n(V);function I(){var e=Object(P["a"])(["\n  query {\n    allTask(sort:CREATED_DESC) {\n      pageInfo {\n        startCursor\n        endCursor\n      }\n      edges {\n        node {\n          ident\n          specId\n          created\n          status\n        }\n      }\n    }\n  }\n"]);return I=function(){return e},e}var M=[{key:"link",label:"Task Id",sortable:!0},{key:"specId",label:"Spec Id",sortable:!0,formatter:function(e){return e||"session"}},{key:"created",label:"Creation Time",sortable:!0},{key:"status",label:"Status"},{key:"actions",label:"actions"}],N=j()(I()),R={name:"Tasks",components:{},apollo:{allTask:N},data:function(){return{searchQuery:"",fields:M,allTask:[],error:"",selected:[]}},computed:{taskList:function(){return this.allTask.edges?this.allTask.edges.map((function(e){return e.node})):[]}},methods:{taskControl:function(e,t){var n=this;"STOP"==e&&l.a.get("".concat(u["a"].API_URL,"/api/admin/task/stop/").concat(t)).then((function(e){n.message=e.data["message"],n.$apollo.queries.allTask.refetch()})).catch((function(e){n.error=e,n.message=n.error,n.$apollo.queries.allTask.refetch()}))}},created:function(){}},D=R,q=(n("655c"),Object(f["a"])(D,x,E,!1,null,null,null)),Z=q.exports,H=n("323e"),U=n.n(H);a["default"].use(g["a"]);var B=[{path:"/",name:"Home",component:_},{path:"/dash",name:"dash",component:function(){return n.e("about").then(n.bind(null,"8e42"))}},{path:"/session/home",name:"SessionHome",component:function(){return n.e("about").then(n.bind(null,"cb05"))}},{path:"/session/view/:uid",component:function(){return n.e("about").then(n.bind(null,"ea4c"))},props:!0},{path:"/episode/view/:uid",component:function(){return n.e("about").then(n.bind(null,"2f76"))},props:function(e){return{uid:e.params.uid,episodeId:e.query.episodeId}}},{path:"/session/compare/",component:function(){return n.e("about").then(n.bind(null,"e013"))},props:function(e){return{uids:e.query.uids}}},{path:"/task/home",name:"TaskHome",component:Z},{path:"/plugin/home/:category?",name:"PluginHome",component:function(){return n.e("about").then(n.bind(null,"3ade"))},props:!0},{path:"/task/new/session_task/:uid?",name:"SessionTask",component:function(){return n.e("about").then(n.bind(null,"a30d"))},props:function(e){return{uid:e.params.uid,agentUid:e.query.agentUid,agentSpecId:e.query.agentSpecId,envSpecId:e.query.envSpecId}}},{path:"/task/new/multiagent/:uid?",name:"MultiAgentSessionTask",component:function(){return n.e("about").then(n.bind(null,"a30d"))},props:function(e){return{uid:e.params.uid,agentUid:e.query.agentUid,agentSpecId:e.query.agentSpecId,envSpecId:e.query.envSpecId}}},{path:"/task/new/:specId",name:"TaskNew",component:function(){return n.e("about").then(n.bind(null,"8f5e"))},props:!0},{path:"/task/specs",name:"TaskSpecs",component:function(){return n.e("about").then(n.bind(null,"3c45"))},props:!0},{path:"/task/view/:uid",name:"TaskView",component:function(){return n.e("about").then(n.bind(null,"b3d1"))},props:!0},{path:"/task/route/:uid",name:"TaskRoute",component:function(){return n.e("about").then(n.bind(null,"de3f"))},props:!0},{path:"/agent/home",name:"AgentHome",component:function(){return n.e("about").then(n.bind(null,"5ad6"))}},{path:"/agent/instances",name:"AgentInstances",component:function(){return n.e("about").then(n.bind(null,"ea73"))},props:!0},{path:"/agent/specs",name:"AgentSpecs",component:function(){return n.e("about").then(n.bind(null,"48e7"))},props:!0},{path:"/agent/view/:uid",name:"AgentView",component:function(){return n.e("about").then(n.bind(null,"89be"))},props:!0},{path:"/agent_spec/:specId",name:"AgentSpecView",component:function(){return n.e("about").then(n.bind(null,"b036"))},props:!0},{path:"/agent/new/:specId",name:"AgentNew",component:function(){return n.e("about").then(n.bind(null,"68c9"))},props:!0},{path:"/env/home",name:"EnvHome",component:function(){return n.e("chunk-2d0cc7f8").then(n.bind(null,"4dd6"))}},{path:"/component_spec/home",name:"ComponentSpecHome",component:function(){return n.e("chunk-34e487bb").then(n.bind(null,"368d"))}},{path:"/component_spec/view/:specId",name:"ComponentSpecView",component:function(){return n.e("about").then(n.bind(null,"7e3c"))},props:!0},{path:"/env_spec/view/:specId",name:"EnvSpecView",component:function(){return n.e("chunk-968b55d2").then(n.bind(null,"c8e4"))},props:!0},{path:"/data_browser",name:"Data Browser",component:function(){return n.e("about").then(n.bind(null,"6906"))},props:function(e){return{path:e.query.path?e.query.path:""}}},{path:"/preferences",name:"Preferences",component:function(){return n.e("about").then(n.bind(null,"a55d"))}},{path:"/help",name:"help",component:O}],z=new g["a"]({routes:B});z.beforeResolve((function(e,t,n){e.name&&U.a.start(),n()})),z.afterEach((function(e,t){U.a.done()}));var $=z,F=n("2f62");a["default"].use(F["a"]);var G=new F["a"].Store({state:{sessions:{},envGroups:{},agents:{},logs:{}},mutations:{ADD_LOG_ENTRY:function(e,t){var n=t["context"];n in e.logs||(e.logs[n]=[]),e.logs[n].push(t["msg"])}},actions:{updateLog:function(e,t){e.commit("ADD_LOG_ENTRY",t)}},getters:{logs:function(e){return function(t){return e.logs[t]}}},modules:{}}),Y=n("4776"),J=n.n(Y),W=(n("b15b"),n("ecee")),K=n("c074"),Q=n("ad3d"),X=n("522d"),ee=n("74ca"),te=n("478e"),ne=n("2bf2"),ae=n("8c60"),oe=Object(te["a"])({uri:u["a"].API_URL+"/graphql"}),re=new ne["a"],se={watchQuery:{fetchPolicy:"cache-and-network",errorPolicy:"ignore"},query:{fetchPolicy:"network-only",errorPolicy:"all"},mutate:{errorPolicy:"all"}},ie=new ee["a"]({link:oe,cache:re,defaultOptions:se}),ce=new X["a"]({defaultClient:ie});a["default"].use(X["a"]),W["c"].add(K["a"]),a["default"].use(J.a),a["default"].use(o["a"]),a["default"].component("font-awesome-icon",Q["a"]),a["default"].use(ae["a"]),a["default"].config.productionTip=!1,new a["default"]({router:$,store:G,data:{},apolloProvider:ce,render:function(e){return e(m)}}).$mount("#app")},cf92:function(e,t,n){"use strict";n.d(t,"a",(function(){return o})),n.d(t,"b",(function(){return r}));n("d3b7");var a=Object({NODE_ENV:"production",BASE_URL:"/"}).VUE_APP_BACKEND_URL;"undefined"===typeof a&&(a="");var o={API_URL:a};function r(){return fetch(a+"/api/config").then((function(e){return e.json()}))}}});
//# sourceMappingURL=app.959e1e49.js.map