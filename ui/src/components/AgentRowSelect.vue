<template>
<div v-if="agent && agent.ident">
<b-row>
    <b-col cols=2>
                        <img :src="`/img/agent_cons/${getImageId(agent.ident)}.SVG`" alt="Image" style="max-height:60px;" class="svgagent"/>

    </b-col>
    <b-col>
        <b-row>
            <b-col>
                <div>
                    <router-link target="_blank" :to="`/agent/view/${agent.ident}`"> {{ agent.ident }}</router-link>
                </div>
            </b-col>

            <b-col>
                <div>
                    <b-button size="sm" variant="primary" @click="select()">Select</b-button>

                </div>
            </b-col>
        </b-row>
        <div class="mt-1"></div>
        <b-row>
            <b-col>

                <div>
                    <span class="data_label mt-1">Spec Id: </span>
                    <span>{{agent.specId}}</span>
                </div>
                <div>
                    <span class="data_label mt-1">created: </span>
                    <span>{{agent.created}}</span>
                </div>
                    </b-col>
    <b-col>
      
            </b-col>


        </b-row>
           <b-row class="mt-2">
                <b-col >
                    <div>
                        <span class="data_label mt-1">Observation Space </span>
                        <pre>{{JSON.stringify(config.observation_space,null,2)}}</pre>
                    </div>
                </b-col>
                <b-col >
                    <div>
                        <span class="data_label mt-1">Action Space </span>
                        <pre>{{JSON.stringify(config.action_space,null,2)}}</pre>
                    </div>
                </b-col>
            </b-row>
        </b-col>
        </b-row>

</div>
</template>

<script >
export default {
    props: {
        agent: Object
    },
    data() {
        return {

        };
    },
    mounted() {
        //
    },
    methods: {
        //

        getImageId(uid) {
            let id = parseInt(uid.split("-")[1]);
            return id % 19;
        },
        select(){
            this.$emit('click')

        }

    },
    computed: {
        lastCheckpoint(){
            if (this.agent && this.agent.lastCheckpoint)
                return JSON.parse(this.agent.lastCheckpoint).id
            else
                return "NA"
        },
        //
        config() {
            return JSON.parse(this.agent.config)
        },
        meta() {
            return JSON.parse(this.agent.meta)
        }

    },
    // Fetches posts when the component is created.
    created() {
        //

    },
    beforeDestroy() {
        //

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->

<style scoped lang="scss">

</style>
