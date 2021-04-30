<template>
<div v-if="item.ident">

    <b-card no-body class="overflow-hidden card-shadow">
        <b-row no-gutters>
            <b-col md="4">
                <b-card-img :src="`${appConfig.API_URL}/api/env_preview_image/${item.environment.ident}`" alt="" style="max-height:350px; max-width:400px; min-height:200px; min-width:250px; height:auto; width: auto;"></b-card-img>
            </b-col>
            <b-col md="8">
                <b-card-body :title="item.ident">
                    <b-card-text>
                        <div class="mt-auto">

                            <div class="data_label">Environment: {{ item.environment.ident }}</div>
                            <div class="data_label">Env Type: {{ item.envType }}</div>
                            <div class="data_label">Observation Space:
                                <span v-if="meta && meta.observation_space">
                                    {{meta.observation_space.class_name}} (args={{meta.observation_space.args}},{{meta.observation_space.kwargs}})
                                </span>
                                <!-- <span v-if="meta && meta.observation_spaces">

                                    {{meta.observation_spaces}})

                                </span> -->
                                <span v-else>NA</span>

                            </div>
                            <div class="data_label">Action Space:
                                <span v-if="meta && meta.action_space">
                                    {{meta.action_space.class_name}} (args={{meta.action_space.args}},{{meta.action_space.kargs}})
                                </span>
                                <!-- <span v-if="meta && meta.action_spaces">

                                    {{meta.action_spaces}})

                                </span> -->
                                <span v-else>NA</span>
                            </div>
<!-- 
                            <div class="data_label" v-if="meta.agents">Agents:
                                {{meta.agents}}
                            </div> -->
                            <div class="data_label" v-if="meta.num_agents">Num Agents:
                                {{meta.num_agents}}
                            </div>

                        </div>
                    </b-card-text>
                </b-card-body>
            </b-col>
        </b-row>
    </b-card>

</div>
</template>

<script>
import {
    appConfig
} from "../app.config";
export default {
    props: {
        item: Object
    },
    data() {
        return {
            appConfig

        };
    },
    mounted() {
        //
    },
    methods: {
        //

    },
    computed: {

        meta() {
            return JSON.parse(this.item.meta)
        }
        //

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

<style>

</style>
