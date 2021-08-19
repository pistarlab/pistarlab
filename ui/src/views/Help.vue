<template>
<div >
    <h1><i class="fa fa-question-circle"></i> Help</h1>
    <b-container fluid>
        <b-row>
            <b-col>
                <b-card>
                    <HelpInfo :fullPage="true" contentId="outline"></HelpInfo>
                </b-card>
            </b-col>
            <b-col cols=4>

                <b-card>

                    <ul>
                        <li>

                            <a target="_blank" href="http://pistarlab.org">Website</a>
                        </li>
                        <li>
                            <a target="_blank" href="http://github.com/pistarlab/pistarlab">Github</a>
                        </li>
                 
                        <li>
                            <a target="_blank" :href="appConfig.API_URL+'/docs/'">Documentation</a>  
                            
                        </li>
                        <li>
                            <a target="_blank" href="TODO">Donate</a>
                        </li>
                    </ul>
                </b-card>
            </b-col>
            
        </b-row>
    </b-container>

</div>
</template>

<script>
export default {
    name: 'Help',
    components: {}
}
</script>
