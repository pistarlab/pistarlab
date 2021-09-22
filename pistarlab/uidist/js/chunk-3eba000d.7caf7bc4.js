(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-3eba000d"],{2532:function(t,e,n){"use strict";var i=n("23e7"),s=n("5a34"),a=n("1d80"),o=n("ab13");i({target:"String",proto:!0,forced:!o("includes")},{includes:function(t){return!!~String(a(this)).indexOf(s(t),arguments.length>1?arguments[1]:void 0)}})},"3ca3":function(t,e,n){"use strict";var i=n("6547").charAt,s=n("69f3"),a=n("7dd0"),o="String Iterator",r=s.set,c=s.getterFor(o);a(String,"String",(function(t){r(this,{type:o,string:String(t),index:0})}),(function(){var t,e=c(this),n=e.string,s=e.index;return s>=n.length?{value:void 0,done:!0}:(t=i(n,s),e.index+=t.length,{value:t,done:!1})}))},"4dd6":function(t,e,n){"use strict";n.r(e);var i=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"page"},[n("div",{staticClass:"page-content"},[t._m(0),n("div",{staticClass:"mt-4"}),n("b-modal",{attrs:{id:"modal-addcustom"}},[t._v(" TODO ")]),n("div",{staticClass:"mt-4"}),n("b-modal",{attrs:{id:"modal-selected",footerClass:"p-2 border-top-0",title:""+t.selectedEnvironment.displayedName,size:"lg",scrollable:"","hide-footer":""}},[n("b-container",{attrs:{fluid:""}},[n("b-row",[n("b-col",[n("div",[n("span",{staticClass:"data_label  mt-2"},[t._v("Categories: ")]),n("span",{staticClass:"data_label"},[t._v(t._s(t.selectedEnvironment.categories))])]),n("div",[n("span",{staticClass:"data_label  mt-2"},[t._v("Version: ")]),n("span",{staticClass:"data_label"},[t._v(t._s(t.selectedEnvironment.version))])]),n("div",[n("span",{staticClass:"data_label  mt-2"},[t._v("Extension: ")]),n("span",{staticClass:"data_label"},[t._v(t._s(t.selectedEnvironment.extensionId)+": "+t._s(t.selectedEnvironment.extensionVersion))])]),n("div",[n("span",{staticClass:"data_label mt-2"},[t._v("Description: ")]),n("span",[t._v(t._s(t.selectedEnvironment.description))])])]),n("b-col",[n("img",{staticStyle:{"max-height":"200px"},attrs:{src:t.appConfig.API_URL+"/api/env_preview_image/env_"+t.selectedEnvironment.ident,alt:""}})])],1)],1),n("div",{staticClass:"mt-4"}),n("h4",[t._v("Environment Specs")]),n("b-container",{attrs:{fluid:""}},t._l(t.selectedEnvironment.specs,(function(e){return n("div",{key:e.ident},[n("b-row",[n("b-col",{staticClass:"text-center  align-middle",attrs:{cols:"3"}},[n("div",{staticClass:"mt-2 mb-2"},[n("router-link",{attrs:{to:"/env_spec/view/"+e.ident}},[n("b-card-img",{staticStyle:{width:"100px"},attrs:{src:t.appConfig.API_URL+"/api/env_preview_image/"+e.ident,alt:"No Image Found"}})],1)],1)]),n("b-col",[n("div",[n("span",[n("router-link",{attrs:{to:"/env_spec/view/"+e.ident}},[t._v(" "+t._s(e.displayedName)+" ")])],1)]),n("div",[n("span",{staticClass:"data_label mt-1"},[t._v("SpecId: ")]),n("span",[t._v(" "+t._s(e.ident)+" ")])]),n("div",[n("span",{staticClass:"data_label mt-1"},[t._v("Type: ")]),n("span",[t._v(t._s(e.envType))])]),n("div",[n("span",{staticClass:"data_label mt-1"},[t._v("Tags: ")]),n("span",[t._v(t._s(e.tags))])]),n("div",[n("span",{staticClass:"data_label mt-1"},[t._v("Description: ")]),n("span",[t._v(t._s(e.description))])])]),n("b-col",{staticClass:"my-auto",attrs:{cols:"2"}},[n("div",[n("b-button",{attrs:{variant:"primary",title:"Assign to agent",to:"/task/new/agenttask/?envSpecId="+e.ident,size:"sm"}},[t._v("Assign")])],1)])],1),n("div",{staticClass:"mt-1"}),n("hr")],1)})),0)],1),n("div",{staticClass:"mt-4"}),n("b-container",{attrs:{fluid:""}},[t.$apollo.queries.environments.loading?n("div",[t._v("Loading..")]):t._e(),n("b-row",[n("b-col",{staticClass:"my-auto"},[n("span",{staticClass:"ml-5 h6"},[t._v(" Collections: ")]),n("b-form-radio-group",{staticClass:"ml-2",attrs:{size:"sm",options:t.collections,buttons:""},model:{value:t.selectedCollection,callback:function(e){t.selectedCollection=e},expression:"selectedCollection"}}),n("b-form-input",{staticClass:"ml-auto",staticStyle:{width:"250px"},attrs:{placeholder:"Search Environments"},model:{value:t.searchtext,callback:function(e){t.searchtext=e},expression:"searchtext"}})],1)],1),n("div",[t.items.length>0?n("div",[n("div",{staticClass:"mt-4"}),n("b-row",[n("b-col",{staticClass:"d-flex flex-wrap justify-content-center  mb-4"},t._l(t.items,(function(e,i){return n("span",{directives:[{name:"b-modal",rawName:"v-b-modal.modal-selected",modifiers:{"modal-selected":!0}}],key:i,staticClass:"m-2",staticStyle:{"min-width":"150px"},on:{click:function(e){return t.selectGroup(i)}}},[n("EnvironmentCard",{attrs:{item:e}})],1)})),0)],1)],1):n("div",{staticClass:"m-5 text-center"},[t._v(" "+t._s(t.message)+" ")])])],1),n("br"),n("div",{staticClass:"mt-4"})],1),n("HelpInfo",{attrs:{contentId:"envs"}})],1)},s=[function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("h1",[n("i",{staticClass:"fa fa-gamepad"}),t._v(" Environments")])}],a=(n("99af"),n("4de4"),n("4160"),n("caad"),n("d81d"),n("d3b7"),n("6062"),n("2532"),n("3ca3"),n("159b"),n("ddb0"),n("4023")),o=function(){var t=this,e=t.$createElement,n=t._self._c||e;return t.item.ident?n("div",[n("div",{staticClass:"h-100 m-1"},[n("div",{staticClass:"text-center image-box mb-3"},[n("img",{staticStyle:{"max-height":"200px"},attrs:{src:t.appConfig.API_URL+"/api/env_preview_image/env_"+t.item.ident,alt:"No Image Found"}})]),n("div",{staticClass:"h4 mb-1"},[t._v(" "+t._s(t.item.displayedName))]),n("div",{staticClass:"small ",staticStyle:{color:"grey"}},[t._v(t._s(t.item.collection))])])]):t._e()},r=[],c={props:{item:Object},data:function(){return{appConfig:a["a"]}},mounted:function(){},methods:{},computed:{},created:function(){},beforeDestroy:function(){}},l=c,d=n("2877"),u=Object(d["a"])(l,o,r,!1,null,null,null),v=u.exports,f=n("bc3a"),p=n.n(f),m=n("39ff"),h={name:"Env",components:{EnvironmentCard:v},apollo:{environments:m["a"]},data:function(){return{environments:[],searchQuery:"",selectedEnvironment:{},message:"No Environments found.",appConfig:a["a"],searchtext:"",selectedCollection:null}},methods:{selectGroup:function(t){this.selectedEnvironment=this.items[t]},createCollections:function(t){var e={};return t.forEach((function(t){var n=[];t.collection in e&&(n=e[t.collection]),n.push(t),e[t.collection]=n})),e},launchHumanMode:function(t){var e=this;this.loading=!0,p.a.get("".concat(a["a"].API_URL,"/api/env/human_mode/").concat(t)).then((function(t){console.log("Run success "+t.data),e.loading=!1})).catch((function(t){console.log("Run error"),e.error=t,e.loading=!1}))}},computed:{allitems:function(){return 0==this.environments.length?[]:this.environments.edges.map((function(t){return t.node})).filter((function(t){return!t.disabled}))},items:function(){var t=this,e=this.allitems;return null!=this.selectedCollection&&(e=e.filter((function(e){return null!=e.collection&&e.collection==t.selectedCollection}))),""!=this.searchtext?e.filter((function(e){return e.displayedName.toLowerCase().includes(t.searchtext.toLowerCase())})):e},collections:function(){var t=new Set;this.allitems.forEach((function(e){""!=e.collection&&null!=e.collection&&t.add(e.collection)}));var e=[];return t.forEach((function(t){e.push({text:t,value:t})})),e.sort((function(t,e){return t.text>e.text})),[{text:"Show All",value:null}].concat(e)}},created:function(){}},_=h,g=Object(d["a"])(_,i,s,!1,null,null,null);e["default"]=g.exports},"5a34":function(t,e,n){var i=n("44e7");t.exports=function(t){if(i(t))throw TypeError("The method doesn't accept regular expressions");return t}},6062:function(t,e,n){"use strict";var i=n("6d61"),s=n("6566");t.exports=i("Set",(function(t){return function(){return t(this,arguments.length?arguments[0]:void 0)}}),s)},6566:function(t,e,n){"use strict";var i=n("9bf2").f,s=n("7c73"),a=n("e2cc"),o=n("0366"),r=n("19aa"),c=n("2266"),l=n("7dd0"),d=n("2626"),u=n("83ab"),v=n("f183").fastKey,f=n("69f3"),p=f.set,m=f.getterFor;t.exports={getConstructor:function(t,e,n,l){var d=t((function(t,i){r(t,d,e),p(t,{type:e,index:s(null),first:void 0,last:void 0,size:0}),u||(t.size=0),void 0!=i&&c(i,t[l],t,n)})),f=m(e),h=function(t,e,n){var i,s,a=f(t),o=_(t,e);return o?o.value=n:(a.last=o={index:s=v(e,!0),key:e,value:n,previous:i=a.last,next:void 0,removed:!1},a.first||(a.first=o),i&&(i.next=o),u?a.size++:t.size++,"F"!==s&&(a.index[s]=o)),t},_=function(t,e){var n,i=f(t),s=v(e);if("F"!==s)return i.index[s];for(n=i.first;n;n=n.next)if(n.key==e)return n};return a(d.prototype,{clear:function(){var t=this,e=f(t),n=e.index,i=e.first;while(i)i.removed=!0,i.previous&&(i.previous=i.previous.next=void 0),delete n[i.index],i=i.next;e.first=e.last=void 0,u?e.size=0:t.size=0},delete:function(t){var e=this,n=f(e),i=_(e,t);if(i){var s=i.next,a=i.previous;delete n.index[i.index],i.removed=!0,a&&(a.next=s),s&&(s.previous=a),n.first==i&&(n.first=s),n.last==i&&(n.last=a),u?n.size--:e.size--}return!!i},forEach:function(t){var e,n=f(this),i=o(t,arguments.length>1?arguments[1]:void 0,3);while(e=e?e.next:n.first){i(e.value,e.key,this);while(e&&e.removed)e=e.previous}},has:function(t){return!!_(this,t)}}),a(d.prototype,n?{get:function(t){var e=_(this,t);return e&&e.value},set:function(t,e){return h(this,0===t?0:t,e)}}:{add:function(t){return h(this,t=0===t?0:t,t)}}),u&&i(d.prototype,"size",{get:function(){return f(this).size}}),d},setStrong:function(t,e,n){var i=e+" Iterator",s=m(e),a=m(i);l(t,e,(function(t,e){p(this,{type:i,target:t,state:s(t),kind:e,last:void 0})}),(function(){var t=a(this),e=t.kind,n=t.last;while(n&&n.removed)n=n.previous;return t.target&&(t.last=n=n?n.next:t.state.first)?"keys"==e?{value:n.key,done:!1}:"values"==e?{value:n.value,done:!1}:{value:[n.key,n.value],done:!1}:(t.target=void 0,{value:void 0,done:!0})}),n?"entries":"values",!n,!0),d(e)}}},"6d61":function(t,e,n){"use strict";var i=n("23e7"),s=n("da84"),a=n("94ca"),o=n("6eeb"),r=n("f183"),c=n("2266"),l=n("19aa"),d=n("861d"),u=n("d039"),v=n("1c7e"),f=n("d44e"),p=n("7156");t.exports=function(t,e,n){var m=-1!==t.indexOf("Map"),h=-1!==t.indexOf("Weak"),_=m?"set":"add",g=s[t],b=g&&g.prototype,x=g,C={},y=function(t){var e=b[t];o(b,t,"add"==t?function(t){return e.call(this,0===t?0:t),this}:"delete"==t?function(t){return!(h&&!d(t))&&e.call(this,0===t?0:t)}:"get"==t?function(t){return h&&!d(t)?void 0:e.call(this,0===t?0:t)}:"has"==t?function(t){return!(h&&!d(t))&&e.call(this,0===t?0:t)}:function(t,n){return e.call(this,0===t?0:t,n),this})};if(a(t,"function"!=typeof g||!(h||b.forEach&&!u((function(){(new g).entries().next()})))))x=n.getConstructor(e,t,m,_),r.REQUIRED=!0;else if(a(t,!0)){var w=new x,E=w[_](h?{}:-0,1)!=w,S=u((function(){w.has(1)})),k=v((function(t){new g(t)})),I=!h&&u((function(){var t=new g,e=5;while(e--)t[_](e,e);return!t.has(-0)}));k||(x=e((function(e,n){l(e,x,t);var i=p(new g,e,x);return void 0!=n&&c(n,i[_],i,m),i})),x.prototype=b,b.constructor=x),(S||I)&&(y("delete"),y("has"),m&&y("get")),(I||E)&&y(_),h&&b.clear&&delete b.clear}return C[t]=x,i({global:!0,forced:x!=g},C),f(x,t),h||n.setStrong(x,t,m),x}},7156:function(t,e,n){var i=n("861d"),s=n("d2bb");t.exports=function(t,e,n){var a,o;return s&&"function"==typeof(a=e.constructor)&&a!==n&&i(o=a.prototype)&&o!==n.prototype&&s(t,o),t}},ab13:function(t,e,n){var i=n("b622"),s=i("match");t.exports=function(t){var e=/./;try{"/./"[t](e)}catch(n){try{return e[s]=!1,"/./"[t](e)}catch(i){}}return!1}},caad:function(t,e,n){"use strict";var i=n("23e7"),s=n("4d64").includes,a=n("44d2"),o=n("ae40"),r=o("indexOf",{ACCESSORS:!0,1:0});i({target:"Array",proto:!0,forced:!r},{includes:function(t){return s(this,t,arguments.length>1?arguments[1]:void 0)}}),a("includes")}}]);
//# sourceMappingURL=chunk-3eba000d.7caf7bc4.js.map