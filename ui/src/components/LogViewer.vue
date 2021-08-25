<template>
<div style="overflow-x: auto;">
    <div v-if="nocard">
        <b-button title="refresh and move to latest output" pill class="mb-3" size="sm" @click="setupEventStream()"><i class="fas fa-sync"></i></b-button>
        <pre :style="'max-height:' + height +';overflow-y: scroll;'" @onchange="updatescroll()" id="rootlogconsole" :class="logClass">{{logdataoutput}}</pre>
    </div>
    <div v-else>
        <b-card :title="title">

            <b-button title="refresh and move to latest output" pill class="mb-3" size="sm" @click="setupEventStream()"><i class="fas fa-sync"></i></b-button>
            <div :style="'max-height:' + height +';overflow-y: scroll;'">
                <pre @onchange="updatescroll()" id="rootlogconsole" :class="logClass">{{logdataoutput}}</pre>
            </div>
        </b-card>
    </div>
</div>
</template>

<script>
export default {
    name: "LogViewer",
    components: {},

    data() {
        return {
            logdataoutput: "_",
            error: "",
            logdata: [],
            curIdx: 0,
            maxSize: 400,
        };
    },
    props: {
        logStreamUrl: String,
        logClass: String,
        title: String,
        nocard: Boolean,
        height: {
            type: String,
            default: "300px"
        }
    },
    computed: {
        //
    },
    methods: {
        renderLogs() {
            var reordered = []
            var startIdx = Math.max(this.curIdx - this.maxSize, 0)
            for (var i = startIdx; i < this.curIdx; i++) {
                reordered.push(this.logdata[i % this.maxSize])
            }
            this.logdataoutput = reordered
                .map((v, k) => v.trim())
                .join("\n");
            setTimeout(this.updatescroll, 0.1)

        },
        setupEventStream() {
            if (this.es) {
                this.es.close()
            }
            console.log("Setup Log Events");
            this.es = new EventSource(`${this.logStreamUrl}`);

            this.es.addEventListener(
                "message",
                (event) => {
                    let logsUpdated = false
                    let info = JSON.parse(event.data);
                    let data_batch = info['entries']
                    let clear_old = info['clear_old']
                    if (clear_old) {
                        this.logdata = []
                        this.curIdx = 0
                    }

                    data_batch.forEach((data) => {
                        this.logdata[this.curIdx % this.maxSize] = data
                        this.curIdx++;
                        logsUpdated = true;

                    })

                    if (logsUpdated) this.renderLogs();

                },
                false
            );
            this.es.onerror = (e) => {
                console.log("An error occurred while attempting to connect. " + e);
                // this.es.close()
            };

        },
        updatescroll() {
            var el = this.$el.querySelector("#rootlogconsole");
            el.scrollIntoView(false)
            el.scrollTop = el.scrollHeight;
        },

    },
    created() {
        console.log(this.logStreamUrl);
        this.setupEventStream()
    },
    beforeDestroy() {
        if (this.es) {
            this.es.close()
        }
    },
};
</script>

<style>
.logdark {
    overflow-y: scroll;
    white-space: pre-wrap;
    background-color: #111;
    color: white;

}

.logdefault {
    overflow-y: scroll;
    white-space: pre-wrap;

    color: black;
}


/* pre .log {
    font-size: 0.7em;
    background-color: #111 !important;
} */


</style>
