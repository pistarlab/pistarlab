<template>
<div v-if="agent && agent.ident">
    <b-row>
        <b-col cols=2 class="text-center">
            <img :src="`/img/agent_spec_icons/agent_${getImageId(agent.specId)}.png`" alt="Image" style="max-height:60px;" />

        </b-col>
        <b-col>
            <b-row>
                <b-col>
                    <div>
                        <router-link target="_blank" :to="`/agent/view/${agent.ident}`"> {{ agent.ident }} <span v-if="agent.name">({{agent.name}})</span></router-link>
                    </div>
                </b-col>

                <b-col cols=2>
                    <div>
                        <b-button size="sm" variant="primary" @click="select()">Select</b-button>

                    </div>
                </b-col>
            </b-row>
            <div class="mt-1"></div>
            <b-row class="small">
                <b-col>

                    <div>
                        <span class="data_label mt-1">Spec Id: </span>
                        <span>{{agent.specId}}</span>
                    </div>
                    <div>
                        <span class="data_label mt-1">created: </span>
                        <span>{{agent.created}}</span>
                    </div>
                                <div>
                        <span class="data_label mt-1">Tags: </span>
                        <span>{{agent.tags.edges.map((n)=> n.node.tagId).join(", ")}}</span>
                    </div>
                    <div>
                        <span class="data_label mt-1">Notes: </span>
                        <span>{{agent.notes}}</span>
                    </div>
                </b-col>

                <b-col>
                    <span v-if="agent.configParsed.interfaces.run.auto_config_spaces">
                       <span class="data_label mt-1">Observation Space </span> assigned at runtime.
                       <br/>
                       <span class="data_label mt-1">Action Space: </span> assigned at runtime.
                    </span>
                    <div v-else>
                        <div>
                            <span class="data_label mt-1">Observation Space: </span>
                            <span>

                                <SpaceInfo :space="agent.configParsed.interfaces.run.observation_space">
                                </SpaceInfo>

                            </span>
                        </div>
                        <div>
                            <span class="data_label mt-1">Action Space: </span>
                            <span>
                                <SpaceInfo :space="agent.configParsed.interfaces.run.action_space">
                                </SpaceInfo>
                            </span>
                        </div>

                    </div>
                </b-col>
            </b-row>
            <!-- <b-row class="mt-2 small">
                <b-col>
                    <div>
                        <span class="data_label mt-1">Observation Space </span>
                        <span v-if="config.observation_space">{{JSON.stringify(config.observation_space,null,2)}}</span>
                        <span v-else>Not Defined</span>
                    </div>
                </b-col>
                <b-col>
                    <div>
                        <span class="data_label mt-1">Action Space </span>
                       <span v-if="config.action_space"> {{JSON.stringify(config.action_space,null,2)}}</span>
                       <span v-else>Not Defined</span>
                    </div>
                </b-col>
            </b-row> -->
        </b-col>
    </b-row>

</div>
</template>

<script>

import SpaceInfo from "./SpaceInfo.vue";
export default {
    components:{
    SpaceInfo
    },
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

        select() {
            this.$emit('click')

        }

    },
    computed: {
        lastCheckpoint() {
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
